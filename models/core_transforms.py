import torch
from torch.nn import Linear


class BaseLinearTransformSTNkD(torch.nn.Module):
    """STN which learns a k-dimensional linear transformation

    Arguments:
        nn (torch.nn.Module) -- module which takes feat_x as input and regresses it to a global feature used to calculate the transform
        nn_feat_size -- the size of the global feature
        k -- the size of trans_x
        batch_size -- the number of examples per batch
    """

    def __init__(self, nn, nn_feat_size, k=3, batch_size=1):
        super().__init__()

        self.nn = nn
        self.k = k
        self.batch_size = batch_size

        # fully connected layer to regress the global feature to a k-d linear transform
        # the transform is initialized to the identity
        self.fc_layer = Linear(nn_feat_size, k * k)
        torch.nn.init.constant_(self.fc_layer.weight, 0)
        torch.nn.init.constant_(self.fc_layer.bias, 0)
        self.identity = torch.eye(k).view(1, k * k).repeat(batch_size, 1)

    def forward(self, feat_x, trans_x, batch):
        """
            Learns and applies a linear transformation to trans_x based on feat_x.
            feat_x and trans_x may be the same or different.
        """
        global_feature = self.nn(feat_x, batch)
        trans = self.fc_layer(global_feature)

        # needed so that transform is initialized to identity
        trans = trans + self.identity.to(feat_x.device)
        trans = trans.view(-1, self.k, self.k)
        self.trans = trans

        # convert trans_x from (N, K) to (B, N, K) to do batched matrix multiplication
        # batch_x = trans_x.view(self.batch_size, -1, trans_x.shape[1])
        batch_x = trans_x.view(trans_x.shape[0], 1, trans_x.shape[1])
        x_transformed = torch.bmm(batch_x, trans[batch])

        return x_transformed.view(len(trans_x), trans_x.shape[1])

    def get_orthogonal_regularization_loss(self):
        loss = torch.mean(
            torch.norm(
                torch.bmm(self.trans, self.trans.transpose(2, 1))
                - self.identity.to(self.trans.device).view(-1, self.k, self.k),
                dim=(1, 2),
            )
        )

        return loss
