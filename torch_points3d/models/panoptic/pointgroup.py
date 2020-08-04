import torch
import os
from torch_points_kernels import region_grow
from torch_geometric.data import Data
from torch_scatter import scatter
import random

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from torch_points3d.core.losses import offset_loss, instance_iou_loss
from .structures import PanopticLabels, PanopticResults
from torch_points3d.utils import is_list


class PointGroup(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroup, self).__init__(option)
        backbone_options = option.get("backbone", {"architecture": "unet"})
        self.Backbone = Minkowski(
            backbone_options.get("architecture", "unet"),
            input_nc=dataset.feature_dimension,
            num_layers=4,
            config=backbone_options.get("config", None),
        )
        self.BackboneHead = Seq().append(FastBatchNorm1d(self.Backbone.output_nc)).append(torch.nn.ReLU())

        self._scorer_is_encoder = option.scorer.architecture == "encoder"
        self._activate_scorer = option.scorer.activate
        self.Scorer = Minkowski(
            option.scorer.architecture, input_nc=self.Backbone.output_nc, num_layers=option.scorer.depth
        )
        self.ScorerHead = Seq().append(torch.nn.Linear(self.Scorer.output_nc, 1)).append(torch.nn.Sigmoid())

        self.Offset = Seq().append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
        self.Offset.append(torch.nn.Linear(self.Backbone.output_nc, 3))

        self.Semantic = (
            Seq()
            .append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
            .append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes))
            .append(torch.nn.LogSoftmax(dim=-1))
        )
        self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss", "semantic_loss", "score_loss"]
        stuff_classes = dataset.stuff_classes
        if is_list(stuff_classes):
            stuff_classes = torch.Tensor(stuff_classes).long()
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), stuff_classes])

    def set_input(self, data, device):
        self.raw_pos = data.pos.to(device)
        self.input = data
        self.labels = data.y.to(device)
        all_labels = {l: data[l].to(device) for l in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels)

    def forward(self, epoch=-1, **kwargs):
        # Backbone
        backbone_features = self.BackboneHead(self.Backbone(self.input).x)

        # Semantic and offset heads
        semantic_logits = self.Semantic(backbone_features)
        offset_logits = self.Offset(backbone_features)

        # Grouping and scoring
        cluster_scores = None
        all_clusters = None
        cluster_type = None
        if epoch == -1 or epoch > self.opt.prepare_epoch:  # Active by default
            all_clusters, cluster_type = self._cluster(semantic_logits, offset_logits)
            if len(all_clusters):
                cluster_scores = self._compute_score(all_clusters, backbone_features, semantic_logits)

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        with torch.no_grad():
            self._dump_visuals(epoch)

        # Compute loss
        self._compute_loss()

    def _cluster(self, semantic_logits, offset_logits):
        """ Compute clusters from positions and votes """
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_pos = region_grow(
            self.raw_pos,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
        )
        clusters_votes = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
        )

        all_clusters = clusters_pos + clusters_votes
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        cluster_type[len(clusters_pos) :] = 1
        return all_clusters, cluster_type

    def _compute_score(self, all_clusters, backbone_features, semantic_logits):
        """ Score the clusters """
        if self._activate_scorer:
            x = []
            coords = []
            batch = []
            for i, cluster in enumerate(all_clusters):
                x.append(backbone_features[cluster])
                coords.append(self.input.coords[cluster])
                batch.append(i * torch.ones(cluster.shape[0]))
            batch_cluster = Data(x=torch.cat(x).cpu(), coords=torch.cat(coords).cpu(), batch=torch.cat(batch).cpu(),)
            score_backbone_out = self.Scorer(batch_cluster)
            if self._scorer_is_encoder:
                cluster_feats = score_backbone_out.x
            else:
                cluster_feats = scatter(
                    score_backbone_out.x, score_backbone_out.batch.long().to(self.device), dim=0, reduce="max"
                )
            cluster_scores = self.ScorerHead(cluster_feats).squeeze(-1)
        else:
            # Use semantic certainty as cluster confidence
            with torch.no_grad():
                cluster_semantic = []
                batch = []
                for i, cluster in enumerate(all_clusters):
                    cluster_semantic.append(semantic_logits[cluster, :])
                    batch.append(i * torch.ones(cluster.shape[0]))
                cluster_semantic = torch.cat(cluster_semantic)
                batch = torch.cat(batch)
                cluster_semantic = scatter(cluster_semantic, batch.long().to(self.device), dim=0, reduce="mean")
                cluster_scores = torch.max(cluster_semantic, 1)[0]
        return cluster_scores

    def _compute_loss(self):
        # Semantic loss
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, self.labels.y, ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss

        # Offset loss
        self.input.instance_mask = self.input.instance_mask.to(self.device)
        self.input.vote_label = self.input.vote_label.to(self.device)
        offset_losses = offset_loss(
            self.output.offset_logits[self.input.instance_mask],
            self.input.vote_label[self.input.instance_mask],
            torch.sum(self.input.instance_mask),
        )
        for loss_name, loss in offset_losses.items():
            setattr(self, loss_name, loss)
            self.loss += self.opt.loss_weights[loss_name] * loss

        # Score loss
        if self.output.cluster_scores is not None and self._activate_scorer:
            self.score_loss = instance_iou_loss(
                self.output.clusters,
                self.output.cluster_scores,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
                min_iou_threshold=self.opt.min_iou_threshold,
                max_iou_threshold=self.opt.max_iou_threshold,
            )
            self.loss += self.score_loss * self.opt.loss_weights["score_loss"]

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
            nms_idx = self.output.get_instances()
            if self.output.clusters is not None:
                data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
                data_visual.cluster_type = self.output.cluster_type[nms_idx]
            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
