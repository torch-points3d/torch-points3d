import logging
import torch
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications import models
import torch_points3d.modules.VoteNet as votenet_module
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class VoteNetModel(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        super(VoteNetModel, self).__init__(option)

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
        proposal_option = option.proposal
        proposal_cls = getattr(votenet_module, proposal_option.module_name)
        self.proposal_cls_module = proposal_cls(
            num_class=proposal_option.num_class,
            vote_aggregation_config=proposal_option.vote_aggregation,
            num_heading_bin=proposal_option.num_heading_bin,
            num_size_cluster=proposal_option.num_size_cluster,
            mean_size_arr=dataset.mean_size_arr,
            num_proposal=proposal_option.num_proposal,
            sampling=proposal_option.sampling,
        )

        # Loss params
        self.loss_params = option.loss_params
        self.loss_params.num_heading_bin = proposal_option.num_heading_bin
        self.loss_params.num_size_cluster = proposal_option.num_size_cluster
        self.loss_params.mean_size_arr = dataset.mean_size_arr.tolist()

        self.losses_has_been_added = False
        self.loss_names = []

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # Forward through backbone model
        self.input = data.to(device)

    def forward(self):
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
        self._compute_losses()

    def _compute_losses(self):
        losses = votenet_module.get_loss(self.input, self.output, self.loss_params)
        for loss_name, loss in losses.items():
            if torch.is_tensor(loss):
                if not self.losses_has_been_added:
                    self.loss_names += [loss_name]
                setattr(self, loss_name, loss)
        self.losses_has_been_added = True

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()
