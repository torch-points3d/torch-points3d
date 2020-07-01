import torch
from torch_points_kernels import region_grow, instance_iou
from torch_geometric.data import Data
import random

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP
from .structures import PanopticLabels, PanopticResults


class PointGroup(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroup, self).__init__(option)
        self.Backbone = Minkowski("unet", input_nc=dataset.feature_dimension, num_layers=4, in_feat=64, in_feat_tr=128)

        self.Scorer = Minkowski("encoder", input_nc=self.Backbone.output_nc, num_layers=4)
        self.ScorerHead = Seq().append(torch.nn.Linear(self.Scorer.output_nc, 1)).append(torch.nn.Sigmoid())

        self.Offset = Seq().append(MLP([[self.Backbone.output_nc, self.Backbone.output_nc]], bias=False))
        self.Offset.append(torch.nn.Linear(self.Backbone.output_nc, 3))
        self.Semantic = (
            Seq().append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes)).append(torch.nn.LogSoftmax())
        )
        self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss", "semantic_loss", "score_loss"]
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), dataset.stuff_classes])

    def set_input(self, data, device):
        self.raw_pos = data.pos.to(device)
        self.input = data
        self.labels = data.y.to(device)
        all_labels = {l: data[l].to(device) for l in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels)

    def forward(self, epoch=-1, **kwargs):
        # Backbone
        backbone_features = self.Backbone(self.input).x

        # Semantic and offset heads
        semantic_logits = self.Semantic(backbone_features)
        offset_logits = self.Offset(backbone_features)

        # Grouping and scoring
        cluster_scores = None
        all_clusters = None
        cluster_type = None
        if epoch == -1 or epoch > self.opt.prepare_epoch:  # Active by default
            predicted_labels = torch.max(semantic_logits, 1)[1]
            clusters_pos = region_grow(
                self.raw_pos.to(self.device),
                predicted_labels,
                self.input.batch.to(self.device),
                ignore_labels=self._stuff_classes.to(self.device),
                radius=self.opt.cluster_radius_search,
            )
            clusters_votes = region_grow(
                self.raw_pos.to(self.device) + offset_logits,
                predicted_labels,
                self.input.batch.to(self.device),
                ignore_labels=self._stuff_classes.to(self.device),
                radius=self.opt.cluster_radius_search,
            )

            all_clusters = clusters_pos + clusters_votes
            all_clusters = [c.to(self.device) for c in all_clusters]
            cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
            cluster_type[len(clusters_pos) :] = 1

            if len(all_clusters):
                x = []
                coords = []
                batch = []
                for i, cluster in enumerate(all_clusters):
                    x.append(backbone_features[cluster])
                    coords.append(self.input.coords[cluster])
                    batch.append(i * torch.ones(cluster.shape[0]))
                batch_cluster = Data(x=torch.cat(x).cpu(), coords=torch.cat(coords).cpu(), batch=torch.cat(batch).cpu())
                cluster_scores = self.ScorerHead(self.Scorer(batch_cluster).x)

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        self._dump_visuals(epoch)

        # Compute loss
        self._compute_loss()

    def _compute_loss(self):
        # Semantic loss
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, self.labels.y, ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss

        # Offset loss
        offset_losses = self._offset_loss(self.labels, self.output)

        # Score loss
        if self.output.cluster_scores is not None:
            ious = instance_iou(
                self.output.clusters, self.labels.instance_labels.to(self.device), self.input.batch.to(self.device)
            ).max(1)[0]
            lower_mask = ious < self.opt.min_iou_threshold
            higher_mask = ious > self.opt.max_iou_threshold
            middle_mask = torch.logical_and(torch.logical_not(lower_mask), torch.logical_not(higher_mask))
            assert torch.sum(lower_mask + higher_mask + middle_mask) == ious.shape[0]
            shat = torch.zeros_like(ious)
            iou_middle = ious[middle_mask]
            shat[higher_mask] = 1
            shat[middle_mask] = (iou_middle - self.opt.min_iou_threshold) / (
                self.opt.max_iou_threshold - self.opt.min_iou_threshold
            )
            self.score_loss = torch.nn.functional.binary_cross_entropy(self.output.cluster_scores, shat)
            self.loss += self.score_loss * self.opt.loss_weights["score_loss"]

        for loss_name, loss in offset_losses.items():
            setattr(self, loss_name, loss)
            self.loss += self.opt.loss_weights[loss_name] * loss

    @staticmethod
    def _offset_loss(data_labels: PanopticLabels, result: PanopticResults):
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

    def _dump_visuals(self, epoch):
        if random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.vote = self.output.offset_logits
            if self.output.clusters is not None:
                data_visual.clusters = [c.cpu() for c in self.output.clusters]
                data_visual.cluster_type = self.output.cluster_type

            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
