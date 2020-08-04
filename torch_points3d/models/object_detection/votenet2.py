import torch
from torch_geometric.data import Data
import os

from torch_points3d.datasets.object_detection.box_data import BoxData
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.votenet import VoteNet
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications import models
import torch_points3d.modules.VoteNet as votenet_module
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from torch_points3d.core.data_transform import AddOnes
from torch_points3d.modules.VoteNet.dense_samplers import RandomSamplerToDense, FPSSamplerToDense


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
        self._dataset = dataset
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
        dropout = option.get("dropout", None)
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        # 2 - SEGMENTATION HEAD
        semantic_supervision = option.get("semantic_supervision", False)
        if semantic_supervision:
            self.Semantic = (
                Seq()
                .append(MLP([self.backbone_model.output_nc, self.backbone_model.output_nc], bias=False))
                .append(torch.nn.Linear(self.backbone_model.output_nc, dataset.num_classes))
                .append(torch.nn.LogSoftmax())
            )
        else:
            self.Semantic = None

        # 3 - CREATE VOTING MODEL
        voting_option = option.voting
        self._num_seeds = voting_option.num_points_to_sample
        voting_cls = getattr(votenet_module, voting_option.module_name)
        self.voting_module = voting_cls(
            vote_factor=voting_option.vote_factor, seed_feature_dim=self.backbone_model.output_nc
        )

        # 4 - CREATE PROPOSAL MODULE
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
        self.semantic_labels = data.y.flatten().to(self.device)
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
        if self.dropout:
            data_features.x = self.dropout(data_features.x)
        data_seeds, seed_inds = self._select_seeds(data_features)

        # Semantic prediction only if full Unet
        semantic_logits = None
        if self.Semantic:
            backbone_feats = data_features.x.clone()
            if backbone_feats.dim() == 3:
                backbone_feats = backbone_feats.transpose(2, 1)
                backbone_feats = backbone_feats.reshape(-1, backbone_feats.shape[2])
            if backbone_feats.shape[0] == self.semantic_labels.shape[0]:
                semantic_logits = self.Semantic(backbone_feats)

        # Box prediction
        data_votes = self.voting_module(data_seeds)
        setattr(data_votes, "seed_inds", seed_inds)  # [B,num_seeds]

        outputs: votenet_module.VoteNetResults = self.proposal_cls_module(data_votes.to(self.device))

        # Set output and compute losses
        self.input = self.input.to(self.device)
        self._extract_gt_center(self.input, outputs)
        self.output = outputs

        # Sets visual data for debugging
        with torch.no_grad():
            self._dump_visuals()

        self._compute_losses(semantic_logits)

    def _select_seeds(self, data_features):
        sampling_id_key = "sampling_id_0"
        if hasattr(data_features, sampling_id_key):
            seed_inds = getattr(data_features, sampling_id_key, None)[:, : self._num_seeds]
            if data_features.pos.shape[1] != self._num_seeds:
                pos = torch.gather(
                    data_features.pos, 1, seed_inds.unsqueeze(-1).repeat(1, 1, data_features.pos.shape[-1])
                )
                x = torch.gather(data_features.x, 2, seed_inds.unsqueeze(1).repeat(1, data_features.x.shape[1], 1))
                data_out = Data(pos=pos, x=x)
            else:
                data_out = Data(pos=data_features.pos, x=data_features.x)
        else:
            sampler = FPSSamplerToDense(num_to_sample=self._num_seeds)
            data_out, seed_inds = sampler.sample(data_features, self._n_batches, self.conv_type)
        return data_out, seed_inds

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()

    def _extract_gt_center(self, data, outputs):
        if self.is_dense_format:
            gt_center = data.center_label[:, :, 0:3]
        else:
            gt_center = data.center_label[:, 0:3].view((self._n_batches, -1, 3))
        outputs.assign_objects(gt_center, self.loss_params.near_threshold, self.loss_params.far_threshold)

    def _compute_losses(self, semantic_logits=None):
        losses = votenet_module.get_loss(self.input, self.output, self.loss_params)
        for loss_name, loss in losses.items():
            if torch.is_tensor(loss):
                if not self.losses_has_been_added:
                    self.loss_names += [loss_name]
                setattr(self, loss_name, loss)

        if semantic_logits is not None:
            if not self.losses_has_been_added:
                self.loss_names += ["semantic_loss"]
            self.semantic_loss = torch.nn.functional.nll_loss(
                semantic_logits, self.semantic_labels, ignore_index=IGNORE_LABEL
            )
            self.loss += 10 * self.semantic_loss
        self.losses_has_been_added = True

    def get_spatial_ops(self):
        return self.backbone_model.get_spatial_ops()

    def _dump_visuals(self):
        if True:
            return
        if not hasattr(self, "visual_count"):
            self.visual_count = 0

        pred_boxes = self.output.get_boxes(self._dataset, apply_nms=True)
        gt_boxes = []

        for idx in range(len(pred_boxes)):
            # Ground truth
            sample_boxes = self.input.instance_box_corners[idx]
            sample_boxes = sample_boxes[self.input.box_label_mask[idx]]
            sample_labels = self.input.sem_cls_label[idx]
            gt_box_data = [BoxData(sample_labels[i].item(), sample_boxes[i]) for i in range(len(sample_boxes))]
            gt_boxes.append(gt_box_data)

        data_visual = Data(pos=self.input.pos, batch=self.input.batch, gt_boxes=gt_boxes, pred_boxes=pred_boxes)

        if not os.path.exists("viz"):
            os.mkdir("viz")
        torch.save(data_visual.to("cpu"), "viz/data_%i.pt" % (self.visual_count))
        self.visual_count += 1
