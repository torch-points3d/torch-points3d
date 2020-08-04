import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from torch_geometric.data import Data
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications import models
import torch_points3d.modules.VoteNet as votenet_module
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.modelfactory import ModelFactory
from torch_points3d.core.data_transform import AddOnes
from torch_points3d.modules.VoteNet.dense_samplers import RandomSamplerToDense

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/votenet")

log = logging.getLogger(__name__)

MAPPING_BACKBONES_TO_API_NAMES = {"kpconv": "KPConv", "pointnet2": "PointNet2", "rsconv": "RSConv"}


class VoteNet(BaseModel):

    """ Create a VoteNet model based on the architecture proposed in
    https://arxiv.org/abs/1904.09664

    Parameters
    ----------
    original : bool
        If True, it will create VoteNet as defined within the original paper
        Code can be found there: https://github.com/facebookresearch/votenet
    backbone: str
        Supported backbones [pointnet , rsconv, kpconv]
    input_nc : int, optional
        Number of channels for the input
    num_classes : int, optional
        Number of different classes to be defined
    mean_size_arr : []
        If available, provide a prior for the box shapes
    compute_loss : bool, default=False
        Wether to simplify user work and compute losses for them.
        Path VoteNet losses: /torch_points3d/modules/VoteNet/loss_helper.py
    """

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

    def __init__(
        self,
        original: bool = True,
        backbone: str = "pointnet2",
        input_nc: int = None,
        num_classes: int = None,
        mean_size_arr=[],
        compute_loss=False,
        *args,
        **kwargs
    ):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        assert input_nc is not None, "VoteNet requieres input_nc to be defined"
        assert num_classes is not None, "VoteNet requieres num_classes to be defined"
        try:
            self._backbone = MAPPING_BACKBONES_TO_API_NAMES[backbone.lower()]
        except:
            raise Exception("Backbone should be within {}".format(MAPPING_BACKBONES_TO_API_NAMES.keys()))

        if original:
            option = OmegaConf.load(os.path.join(PATH_TO_CONFIG, "votenet.yaml"))
        else:
            option = OmegaConf.load(os.path.join(PATH_TO_CONFIG, "votenet_backbones.yaml"))

        ModelFactory.resolve_model(option, input_nc, kwargs)

        self._original = original
        self._kwargs = kwargs
        self._compute_loss = compute_loss

        super(VoteNet, self).__init__(option)

        # 1 - CREATE BACKBONE MODEL
        if original:
            backbone_option = option.backbone
            backbone_cls = getattr(models, backbone_option.model_type)
            self.backbone_model = backbone_cls(architecture="unet", input_nc=input_nc, config=backbone_option)
        else:
            backbone_cls = getattr(models, self._backbone)
            voting_option = option.voting
            self.backbone_model = backbone_cls(
                architecture="unet",
                input_nc=input_nc,
                num_layers=4,
                output_nc=self._get_attr(voting_option, "feat_dim"),
                **kwargs,
            )
            self._kpconv_backbone = self._backbone == "KPConv"
            self.sampler = RandomSamplerToDense(num_to_sample=voting_option.num_points_to_sample)

        self.conv_type = self.backbone_model.conv_type
        self.is_dense_format = self.conv_type == "DENSE"

        # 2 - CREATE VOTING MODEL
        voting_option = option.voting
        voting_cls = getattr(votenet_module, voting_option.module_name)
        self.voting_module = voting_cls(
            vote_factor=self._get_attr(voting_option, "vote_factor"),
            seed_feature_dim=self._get_attr(voting_option, "feat_dim"),
        )

        # 3 - CREATE PROPOSAL MODULE
        proposal_option = option.proposal
        proposal_cls = getattr(votenet_module, proposal_option.module_name)
        self.proposal_cls_module = proposal_cls(
            num_class=num_classes,
            vote_aggregation_config=proposal_option.vote_aggregation,
            num_heading_bin=proposal_option.num_heading_bin,
            mean_size_arr=mean_size_arr,
            num_proposal=proposal_option.num_proposal,
            sampling=proposal_option.sampling,
        )

        # Loss params
        self.loss_params = option.loss_params
        self.loss_params.num_heading_bin = proposal_option.num_heading_bin
        if isinstance(mean_size_arr, np.ndarray):
            self.loss_params.mean_size_arr = mean_size_arr.tolist()
        else:
            self.loss_params.mean_size_arr = mean_size_arr

        self.losses_has_been_added = False
        self.loss_names = []

    def _set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # Forward through backbone model
        if self.is_dense_format:
            self.input = data
            self._num_batches = data.pos.shape[0]
        else:
            if self._kpconv_backbone:
                data = AddOnes()(data)
                if data.x is not None:
                    data.x = torch.cat([data.x, data.ones.float()], dim=-1)
                else:
                    data.x = data.ones.float()
            self.input = Data(pos=data.pos, x=data.x, batch=data.batch).to(self.device)
            self._num_batches = len(data.id_scan)

    def forward(self, data):
        self._set_input(data)

        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data_features = self.backbone_model.forward(self.input)
        if self._original:
            sampling_id_key = "sampling_id_0"
            num_seeds = data_features.pos.shape[1]
            seed_inds = getattr(data_features, sampling_id_key, None)[:, :num_seeds]
        else:
            data_features, seed_inds = self.sampler.sample(data_features, self._num_batches, self.conv_type)
        data_votes = self.voting_module(data_features)
        setattr(data_votes, "seed_inds", seed_inds)  # [B,num_seeds]

        outputs: votenet_module.VoteNetResults = self.proposal_cls_module(data_votes)

        # Set output and compute losses
        self._extract_gt_center(data, outputs)
        self.input = data
        self.output = outputs
        if self._compute_loss:
            self._compute_losses()

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()

    def _extract_gt_center(self, data, outputs):
        if self.is_dense_format:
            gt_center = data.center_label[:, :, 0:3]
        else:
            gt_center = data.center_label[:, 0:3].view((self._num_batches, -1, 3))
        outputs.assign_objects(gt_center, self.loss_params.near_threshold, self.loss_params.far_threshold)

    def _get_attr(self, opt, arg_name):
        arg = self._kwargs.get(arg_name, None)
        if arg is None:
            arg = opt.get(arg_name, None)
        return arg

    def _compute_losses(self):
        losses = votenet_module.get_loss(self.input, self.output, self.loss_params)
        for loss_name, loss in losses.items():
            if torch.is_tensor(loss):
                if not self.losses_has_been_added:
                    self.loss_names += [loss_name]
                setattr(self, loss_name, loss)
        self.losses_has_been_added = True
