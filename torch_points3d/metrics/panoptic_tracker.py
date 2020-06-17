from typing import Dict, List, Any
import torchnet as tnt
import torch
from collections import OrderedDict

from torch_points3d.models.model_interface import TrackerInterface
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL


class PanopticTracker(SegmentationTracker):
    """ Class that provides tracking of semantic sgmentation as well as 
    instance segmentation """
