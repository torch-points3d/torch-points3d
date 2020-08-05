import logging
import numpy as np
import torch
import os
from torch_geometric.data import Data

from torch_points3d.datasets.object_detection.box_data import BoxData
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications import models
import torch_points3d.modules.VoteNet as votenet_module
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class VoteNetModel(BaseModel):
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
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        super(VoteNetModel, self).__init__(option)
        self._dataset = dataset

        # 1 - CREATE BACKBONE MODEL
        input_nc = dataset.feature_dimension
        backbone_option = option.backbone
        backbone_cls = getattr(models, backbone_option.model_type)
        self.backbone_model = backbone_cls(architecture="unet", input_nc=input_nc, config=backbone_option)

        # 2 - CREATE VOTING MODEL
        voting_option = option.voting
        voting_cls = getattr(votenet_module, voting_option.module_name)
        self.voting_module = voting_cls(vote_factor=voting_option.vote_factor, seed_feature_dim=voting_option.feat_dim)

        # 3 - CREATE PROPOSAL MODULE
        num_classes = dataset.num_classes
        proposal_option = option.proposal
        proposal_cls = getattr(votenet_module, proposal_option.module_name)
        self.proposal_cls_module = proposal_cls(
            num_class=num_classes,
            vote_aggregation_config=proposal_option.vote_aggregation,
            num_heading_bin=proposal_option.num_heading_bin,
            mean_size_arr=dataset.mean_size_arr,
            num_proposal=proposal_option.num_proposal,
            sampling=proposal_option.sampling,
        )

        # Loss params
        self.loss_params = option.loss_params
        self.loss_params.num_heading_bin = proposal_option.num_heading_bin
        mean_size_arr = dataset.mean_size_arr
        if isinstance(mean_size_arr, torch.Tensor):
            mean_size_arr = mean_size_arr.numpy().tolist()
        if isinstance(dataset.mean_size_arr, np.ndarray):
            mean_size_arr = mean_size_arr.tolist()
        self.loss_params.mean_size_arr = mean_size_arr

        self.losses_has_been_added = False
        self.loss_names = []

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # Forward through backbone model
        self.input = data.to(device)

    def forward(self, *args, **kwargs):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data_features = self.backbone_model.forward(self.input)
        data_votes = self.voting_module(data_features)

        sampling_id_key = "sampling_id_0"
        num_seeds = data_features.pos.shape[1]
        seed_inds = getattr(data_features, sampling_id_key, None)[:, :num_seeds]
        setattr(data_votes, "seed_inds", seed_inds)  # [B,num_seeds]
        outputs: votenet_module.VoteNetResults = self.proposal_cls_module(data_votes)

        # Associate proposal and GT objects by point-to-point distances
        gt_center = self.input.center_label[:, :, 0:3]
        outputs.assign_objects(gt_center, self.loss_params.near_threshold, self.loss_params.far_threshold)

        # Set output and compute losses
        self.output = outputs
        with torch.no_grad():
            self._dump_visuals()
        self._compute_losses()

    def _compute_losses(self):
        losses = votenet_module.get_loss(self.input, self.output, self.loss_params)
        for loss_name, loss in losses.items():
            if torch.is_tensor(loss):
                if not self.losses_has_been_added:
                    self.loss_names += [loss_name]
                setattr(self, loss_name, loss)
        self.losses_has_been_added = True

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

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()
