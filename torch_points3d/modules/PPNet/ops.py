import torch
import torch.nn as nn
from torch_points3d.core.common_modules.base_modules import FastBatchNorm1d
from torch_points3d.core.common_modules.gathering import gather


class PosPoolLayer(torch.nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        radius,
        position_embedding="xyz",
        reduction="avg",
        output_conv=False,
        activation=torch.nn.LeakyReLU(negative_slope=0.2),
        bn_momentum=0.02,
        bn=FastBatchNorm1d,
    ):
        super(PosPoolLayer, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.radius = radius
        self.position_embedding = position_embedding
        self.reduction = reduction
        self.output_conv = True if num_outputs != num_inputs else output_conv
        if bn:
            self.bn = bn(num_inputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation
        if self.output_conv:
            self.oconv = torch.nn.Sequential(
                nn.Linear(num_inputs, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum), activation
            )

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        N = query_points.shape[0]
        M = neighbors.shape[1]
        # Add a fake point in the last row for shadow neighbors
        shadow_point = torch.zeros_like(support_points[:1, :])
        support_points = torch.cat([support_points, shadow_point], dim=0)
        # Get neighbor points [N, M, d]
        neighbor_points = gather(support_points, neighbors)
        # Center every neighborhood
        relative_position = neighbor_points - query_points.unsqueeze(1)
        relative_position = relative_position / self.radius

        # Deal with input feature
        shadow_features = torch.zeros_like(x[:1, :])
        support_features = torch.cat([x, shadow_features], dim=0)
        neighborhood_features = gather(support_features, neighbors)

        if self.position_embedding == "xyz":
            geo_prior = relative_position
            mid_fdim = 3
            shared_channels = self.num_inputs // 3
        elif self.position_embedding == "sin_cos":
            position_mat = relative_position  # [N, M, 3]
            if self.num_inputs == 9:
                feat_dim = 1
                wave_length = 1000
                alpha = 100
                feat_range = torch.arange(feat_dim, dtype=x.dtype).to(x.device)  # (feat_dim, )
                dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
                position_mat = alpha * position_mat.unsqueeze(-1)
                div_mat = position_mat / dim_mat  # [N, M, 3, feat_dim]
                sin_mat = torch.sin(div_mat)  # [N, M, 3, feat_dim]
                cos_mat = torch.cos(div_mat)  # [N, M, 3, feat_dim]
                embedding = torch.cat([sin_mat, cos_mat], -1)  # [N, M, 3, 2*feat_dim]
                embedding = embedding.view(N, M, 6)
                embedding = torch.cat([embedding, relative_position], -1)  # [N, M, 9]
                geo_prior = embedding  # [N, M, mid_dim]
            else:
                feat_dim = self.num_inputs // 6
                wave_length = 1000
                alpha = 100
                feat_range = torch.arange(feat_dim, dtype=x.dtype).to(x.device)  # (feat_dim, )
                dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
                position_mat = alpha * position_mat.unsqueeze(-1)  # [N, M, 3, 1]
                div_mat = position_mat / dim_mat  # [N, M, 3, feat_dim]
                sin_mat = torch.sin(div_mat)  # [N, M, 3, feat_dim]
                cos_mat = torch.cos(div_mat)  # [N, M, 3, feat_dim]
                embedding = torch.cat([sin_mat, cos_mat], -1)  # [N, M, 3, 2*feat_dim]
                embedding = embedding.view(N, M, self.num_inputs)  # [N, M, 6*feat_dim]
                geo_prior = embedding  # [N, M, mid_dim]
            mid_fdim = self.num_inputs
            shared_channels = 1
        else:
            raise NotImplementedError
        geo_prior = geo_prior.unsqueeze(-1)

        feature_map = neighborhood_features.view(N, M, mid_fdim, shared_channels)
        aggregation_feature = geo_prior * feature_map
        aggregation_feature = aggregation_feature.view(N, -1, self.num_inputs)  # [N, M, d]

        if self.reduction == "sum":
            aggregation_feature = torch.sum(aggregation_feature, 1)  # [N, d]
        elif self.reduction == "avg":
            aggregation_feature = torch.sum(aggregation_feature, 1)  # [N, d]
            padding_num = torch.max(neighbors)
            neighbors_n = torch.sum((neighbors < padding_num), -1) + 1e-5
            aggregation_feature = aggregation_feature / neighbors_n.unsqueeze(-1)
        elif self.reduction == "max":
            # mask padding
            batch_mask = torch.zeros_like(x)  # [n0_points, d]
            batch_mask = torch.cat([batch_mask, -65535 * torch.ones_like(batch_mask[:1, :])], dim=0)
            batch_mask = gather(batch_mask, neighbors)  # [N, M, d]
            aggregation_feature = aggregation_feature + batch_mask
            aggregation_feature = torch.max(aggregation_feature, 1)  # [N, d]
        else:
            raise NotImplementedError("Reduction {} not supported in PosPool".format(self.reduction))

        if self.bn:
            aggregation_feature = self.bn(aggregation_feature)
        aggregation_feature = self.activation(aggregation_feature)

        if self.output_conv:
            aggregation_feature = self.oconv(aggregation_feature)

        return aggregation_feature
