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
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve
from torch_points3d.core.data_transform import AddOnes

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/votenet")

log = logging.getLogger(__name__)


def resolve_model(model_config, num_features, kwargs):
    """ Parses the model config and evaluates any expression that may contain constants
    Overrides any argument in the `define_constants` with keywords wrgument to the constructor
    """
    # placeholders to subsitute
    constants = {
        "FEAT": max(num_features, 0),
    }

    # user defined contants to subsitute
    if "define_constants" in model_config.keys():
        constants.update(dict(model_config.define_constants))
        define_constants = model_config.define_constants
        for key in define_constants.keys():
            value = kwargs.get(key)
            if value:
                constants[key] = value
    resolve(model_config, constants)


def VoteNet(
    original: bool = False,
    backbone: str = "rsconv",
    input_nc: int = None,
    num_classes: int = None,
    mean_size_arr=[],
    compute_loss=False,
    *args,
    **kwargs
):
    """ Create a VoteNet model with several backbones model
    Parameters
    ----------
    architecture : str, optional
        Architecture of the model, choose from unet, encoder and decoder
    input_nc : int, optional
        Number of channels for the input
    output_nc : int, optional
        If specified, then we add a fully connected head at the end of the network to provide the requested dimension
    num_layers : int, optional
        Depth of the network
    config : DictConfig, optional
        Custom config, overrides the num_layers and architecture parameters
    """

    if original:
        model_config = OmegaConf.load(os.path.join(PATH_TO_CONFIG, "votenet.yaml"))
        resolve_model(model_config, input_nc, kwargs)
        return VoteNetPaper(
            model_config,
            input_nc=input_nc,
            num_classes=num_classes,
            mean_size_arr=mean_size_arr,
            compute_loss=compute_loss,
            *args,
            **kwargs,
        )
    else:
        model_config = OmegaConf.load(os.path.join(PATH_TO_CONFIG, "votenet_backbones.yaml"))
        resolve_model(model_config, input_nc, kwargs)
        mapping_backbones = {"kpconv": "KPConv", "pointnet2": "PointNet2", "rsconv": "RSConv"}
        return VoteNetBackbones(
            model_config,
            backbone=mapping_backbones[backbone.lower()],
            input_nc=input_nc,
            num_classes=num_classes,
            mean_size_arr=mean_size_arr,
            compute_loss=compute_loss,
            *args,
            **kwargs,
        )


class VoteNetBase(BaseModel):
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

    def get_attr(self, opt, arg_name):
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

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()


class VoteNetBackbones(VoteNetBase):
    def sample(self, data):
        if self.conv_type == "DENSE":
            idx = torch.randint(0, data.pos.shape[1], (data.pos.shape[0], self._num_points_to_sample,))
            data.pos = torch.gather(data.pos, 1, idx.unsqueeze(-1).repeat(1, 1, data.pos.shape[-1]))
            data.x = torch.gather(data.x, 2, idx.unsqueeze(1).repeat(1, data.x.shape[1], 1))
            return data, idx
        else:
            pos = []
            x = []
            idx_out = []
            num_points = 0
            for batch_idx in range(self._num_batches):
                batch_mask = data.batch == batch_idx
                pos_masked = data.pos[batch_mask, :]
                x_masked = data.x[batch_mask]
                idx = torch.randint(0, pos_masked.shape[0], (self._num_points_to_sample,))
                pos.append(pos_masked[idx])
                x.append(x_masked[idx])
                idx_out.append(idx + num_points)
                num_points += pos_masked.shape[0]
            data.pos = torch.stack(pos)
            data.x = torch.stack(x).permute(0, 2, 1)
            return data, torch.cat(idx_out, dim=0)

    def __init__(
        self,
        option,
        backbone: str = "rsconv",
        input_nc: int = None,
        num_classes: int = None,
        mean_size_arr=[],
        compute_loss: bool = False,
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
        super(VoteNetBackbones, self).__init__(option)
        self._kwargs = kwargs
        self._compute_loss = compute_loss

        # 1 - CREATE BACKBONE MODEL
        backbone_cls = getattr(models, backbone)
        voting_option = option.voting
        self._kpconv_backbone = backbone == "KPConv"

        self.backbone_model = backbone_cls(
            architecture="unet",
            input_nc=input_nc,
            num_layers=4,
            output_nc=self.get_attr(voting_option, "feat_dim"),
            **kwargs,
        )
        self.conv_type = self.backbone_model.conv_type

        self._num_points_to_sample = voting_option.num_points_to_sample
        # 2 - CREATE VOTING MODEL
        voting_cls = getattr(votenet_module, voting_option.module_name)
        self.voting_module = voting_cls(
            vote_factor=self.get_attr(voting_option, "vote_factor"),
            seed_feature_dim=self.get_attr(voting_option, "feat_dim"),
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
        if self._kpconv_backbone:
            data = AddOnes()(data)
            if data.x is not None:
                data.x = torch.cat([data.x, data.ones.float()], dim=-1)
            else:
                data.x = data.ones.float()
            self.input = Data(pos=data.pos, x=data.x, batch=data.batch).to(self.device)
            self._num_batches = len(data.id_scan)
        else:
            self.input = data

    def forward(self, data):
        self._set_input(data)

        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data_features = self.backbone_model.forward(self.input)
        data_sampled, idx = self.sample(data_features.clone())
        data_votes = self.voting_module(data_sampled)

        setattr(data_votes, "seed_inds", idx)  # [B,num_seeds]
        outputs: votenet_module.VoteNetResults = self.proposal_cls_module(data_votes)

        # Associate proposal and GT objects by point-to-point distances
        if not self._kpconv_backbone:
            gt_center = self.input.center_label[:, :, 0:3]
        else:
            gt_center = data.center_label[:, 0:3].view((self._num_batches, -1, 3))
            self.input = data
        outputs.assign_objects(gt_center, self.loss_params.near_threshold, self.loss_params.far_threshold)

        # Set output and compute losses
        self.output = outputs
        if self._compute_loss:
            self._compute_losses()


class VoteNetPaper(VoteNetBase):
    def __init__(
        self,
        option,
        input_nc: int = None,
        num_classes: int = None,
        mean_size_arr=[],
        compute_loss: bool = False,
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
        super(VoteNetPaper, self).__init__(option)
        self._kwargs = kwargs
        self._compute_loss = compute_loss

        # 1 - CREATE BACKBONE MODEL
        backbone_option = option.backbone
        backbone_cls = getattr(models, backbone_option.model_type)
        self.backbone_model = backbone_cls(architecture="unet", input_nc=input_nc, config=backbone_option)

        # 2 - CREATE VOTING MODEL
        voting_option = option.voting
        voting_cls = getattr(votenet_module, voting_option.module_name)
        self.voting_module = voting_cls(
            vote_factor=self.get_attr(voting_option, "vote_factor"),
            seed_feature_dim=self.get_attr(voting_option, "feat_dim"),
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
        self.input = data.to(self.device)

    def forward(self, data):
        self._set_input(data)

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
        if self._compute_loss:
            self._compute_losses()
