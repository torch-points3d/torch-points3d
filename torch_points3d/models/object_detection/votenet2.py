import torch

from torch_points3d.applications.votenet import VoteNet
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications import models
import torch_points3d.modules.VoteNet as votenet_module
from torch_points3d.core.data_transform import AddOnes
from torch_points3d.core.spatial_ops import RandomSamplerToDense


class VoteNet2(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = [
        "center_label",
        "heading_class_label",
        "heading_residual_label",
        "size_class_label",
        "size_residual_label",
        "sem_cls_label",
        "box_label_mask",
        "vote_label",
        "vote_label_mask",
    ]

    def __init__(self, option, model_type, dataset, modules):
        super(VoteNet2, self).__init__(option)

        # 1 - CREATE BACKBONE MODEL
        input_nc = dataset.feature_dimension
        backbone_option = option.backbone
        backbone_cls = getattr(models, backbone_option.model_type)
        backbone_extr_options = backbone_option.get("extra_options", {})
        self.backbone_model = backbone_cls(
            architecture="unet", input_nc=input_nc, num_layers=4, config=backbone_option.config, **backbone_extr_options
        )
        self._kpconv_backbone = backbone_cls.__name__ == "KPConv"
        self.is_dense_format = self.conv_type == "DENSE"

        # 2 - CREATE VOTING MODEL
        voting_option = option.voting
        self._num_seeds = voting_option.num_points_to_sample
        voting_cls = getattr(votenet_module, voting_option.module_name)
        self.voting_module = voting_cls(
            vote_factor=voting_option.vote_factor, seed_feature_dim=self.backbone_model.output_nc
        )

        # 3 - CREATE PROPOSAL MODULE
        proposal_option = option.proposal
        proposal_option.vote_aggregation.down_conv_nn = [
            [self.backbone_model.output_nc + 3, self.backbone_model.output_nc, self.backbone_model.output_nc,]
        ]
        proposal_cls = getattr(votenet_module, proposal_option.module_name)
        self.proposal_cls_module = proposal_cls(
            num_class=proposal_option.num_class,
            vote_aggregation_config=proposal_option.vote_aggregation,
            num_heading_bin=proposal_option.num_heading_bin,
            mean_size_arr=dataset.mean_size_arr,
            num_proposal=proposal_option.num_proposal,
            sampling=proposal_option.sampling,
        )

        # Loss params
        self.loss_params = option.loss_params
        self.loss_params.num_heading_bin = proposal_option.num_heading_bin
        self.loss_params.mean_size_arr = dataset.mean_size_arr.tolist()

        self.losses_has_been_added = False
        self.loss_names = []

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # Forward through backbone model
        if self.is_dense_format:
            self.input = data
            self._n_batches = data.pos.shape[0]
        else:
            if self._kpconv_backbone:
                data = AddOnes()(data)
                if data.x is not None:
                    data.x = torch.cat([data.x, data.ones.float()], dim=-1)
                else:
                    data.x = data.ones.float()
            self.input = data
            self._n_batches = torch.max(data.batch) + 1

    def forward(self, **kwargs):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data_features = self.backbone_model.forward(self.input)
        sampling_id_key = "sampling_id_0"
        if hasattr(data_features, sampling_id_key):
            seed_inds = getattr(data_features, sampling_id_key, None)[:, : self._num_seeds]
            if data_features.pos.shape[1] != self._num_seeds:
                data_features.pos = torch.gather(
                    data_features.pos, 1, seed_inds.unsqueeze(-1).repeat(1, 1, data_features.pos.shape[-1])
                )
                data_features.x = torch.gather(
                    data_features.x, 2, seed_inds.unsqueeze(1).repeat(1, data_features.x.shape[1], 1)
                )
        else:
            sampler = RandomSamplerToDense(num_to_sample=self._num_seeds)
            data_features, seed_inds = sampler.sample(data_features, self._n_batches, self.conv_type)
        data_votes = self.voting_module(data_features)
        setattr(data_votes, "seed_inds", seed_inds)  # [B,num_seeds]

        outputs: votenet_module.VoteNetResults = self.proposal_cls_module(data_votes.to(self.device))

        # Set output and compute losses
        self.input = self.input.to(self.device)
        self._extract_gt_center(self.input, outputs)
        self.output = outputs
        self._compute_losses()

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()

    def _extract_gt_center(self, data, outputs):
        if self.is_dense_format:
            gt_center = data.center_label[:, :, 0:3]
        else:
            gt_center = data.center_label[:, 0:3].view((self._n_batches, -1, 3))
        outputs.assign_objects(gt_center, self.loss_params.near_threshold, self.loss_params.far_threshold)

    def _compute_losses(self):
        losses = votenet_module.get_loss(self.input, self.output, self.loss_params)
        for loss_name, loss in losses.items():
            if torch.is_tensor(loss):
                if not self.losses_has_been_added:
                    self.loss_names += [loss_name]
                setattr(self, loss_name, loss)
        self.losses_has_been_added = True

    def get_spatial_ops(self):
        return self.backbone_model.get_spatial_ops()
