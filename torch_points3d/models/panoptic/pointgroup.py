import torch
from typing import NamedTuple

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP


class PointGroupResults(NamedTuple):
    semantic_logits: torch.Tensor
    offset_logits: torch.Tensor
    cluster_scores: torch.Tensor


class PointGrouplabels(NamedTuple):
    center_label: torch.Tensor
    y: torch.Tensor
    num_instances: torch.Tensor
    instance_labels: torch.Tensor
    instance_mask: torch.Tensor
    vote_label: torch.Tensor


class PointGroup(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PointGrouplabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroup, self).__init__(option)
        self.Backbone = Minkowski("unet", input_nc=dataset.feature_dimension, num_layers=4)
        # self.Scorer = Minkowski("encoder", input_nc=dataset.feature_dimension, num_layers=2)

        self.Offset = Seq().append(MLP([[self.Backbone.output_nc, self.Backbone.output_nc]], bias=False))
        self.Offset.append(torch.nn.Linear(self.Backbone.output_nc, 3))

        self.Semantic = (
            Seq().append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes)).append(torch.nn.LogSoftmax())
        )

    def set_input(self, data, device):
        self.raw_pos = torch.stack((data.pos_x, data.pos_y, data.pos_z), 0).T.to(device)
        self.input = data
        self.labels = data.y.to(device)
        all_labels = {l: data[l].to(device) for l in self.__REQUIRED_LABELS__}
        self.all_labels = PointGrouplabels(**all_labels)

    def forward(self):
        backbone_features = self.Backbone(self.input).x

        semantic_logits = self.Semantic(backbone_features)
        offset_logits = self.Offset(backbone_features)

        self.output = PointGroupResults(
            semantic_logits=semantic_logits, offset_logits=offset_logits, cluster_scores=None
        )
        self._compute_loss()

    def _compute_loss(self):
        self.loss = self.opt.loss_weights.semantic * torch.nn.functional.nll_loss(
            self.output.semantic_logits, self.labels, ignore_index=IGNORE_LABEL
        )
        offset_losses = self._offset_loss(self.all_labels, self.output)
        for loss_name, loss in offset_losses.items():
            setattr(self, loss_name, loss)
            self.loss += self.opt.loss_weights[loss_name] * loss

    @staticmethod
    def _offset_loss(data_labels: PointGrouplabels, result: PointGroupResults):
        instance_mask = data_labels.instance_mask
        pt_offsets = result.offset_logits[instance_mask, :]

        gt_offsets = data_labels.vote_label[instance_mask, :]
        pt_diff = pt_offsets - gt_offsets
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
        offset_norm_loss = torch.sum(pt_dist) / (torch.sum(instance_mask) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = -(gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff) / (torch.sum(instance_mask) + 1e-6)

        return {"offset_norm_loss": offset_norm_loss, "offset_dir_loss": offset_dir_loss}

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()
