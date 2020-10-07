"""
Implementation of Generative Sparse Detection Networks for 3D Single-shot Object Detection
https://arxiv.org/pdf/2006.12356.pdf
"""
import torch
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
import MinkowskiEngine as ME
import math

from torch_points3d.modules.MinkowskiEngine.api_modules import ResNetDown
from torch_points3d.modules.GSDN.layers import GSDNUp
from torch_points3d.modules.GSDN.gsdn_results import GSDNLayerPrediction, GSDNResults


class GSDN(UnwrappedUnetBasedModel):
    __REQUIRED_DATA__ = ["pos", "x"]
    __REQUIRED_LABELS__ = ["center_label", "size_label", "sem_cls_label", "box_label_mask", "grid_size"]

    def __init__(self, option, model_type, dataset, modules):
        self._dataset = dataset

        backbone_option = option.backbone
        backbone_option.up_conv.num_classes = dataset.num_classes
        backbone_option.up_conv.nb_anchors = len(self.anchors(1))
        super().__init__(backbone_option, model_type, dataset, modules)
        self.loss_names = ["loss", "sparsity_loss", "anchor_loss", "sem_loss", "regr_loss"]

    def set_input(self, data, *args):
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.input = ME.SparseTensor(data.x, coords=coords).to(self.device)
        self.raw_data = data
        self._n_batches = torch.max(data.batch) + 1

    def forward(self, *args, **kwargs):
        data = self.input
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)

        # Middle layer
        data = self.down_modules[-1](data)
        stack_down.append(None)

        # Up convs
        boxes = []
        anchor_size = 1
        for i in range(len(self.up_modules)):
            data, box = self.up_modules[i](data, stack_down.pop())

            # Set box anchors and grid_size
            box.anchors = self.anchors(anchor_size).to(box.device)
            box.grid_size = self.raw_data.grid_size[0].to(box.device)

            # append to hierarchy of predictions
            boxes.append(box)

            # increase anchor_size
            anchor_size *= 2

        self.output = GSDNResults(boxes)
        self.coord_manager = data.coords_man

        if self.raw_data.center_label is not None:
            self._compute_losses()

    def _compute_losses(self):
        # Set labels
        with torch.no_grad():
            centre_labels, size_labels, class_labels = self._extract_gt()
            self.output.evaluate_labels(centre_labels, size_labels, class_labels, self.coord_manager)

        #  Losses
        self.anchor_loss = self.output.get_anchor_loss()
        self.sparsity_loss = self.output.get_sparsity_loss()
        self.sem_loss = self.output.get_semantic_loss()
        self.regr_loss = self.output.get_regression_loss()
        self.loss = self.sparsity_loss + self.anchor_loss + self.sem_loss + 0.1 * self.regr_loss

    def _extract_gt(self):
        """ Simple utility to extract data from the input array and format it as a list of labels, each
        item in the list is one item in the batch
        """
        box_mask = self.raw_data.box_label_mask.reshape(self._n_batches, -1).to(self.device)
        centre_labels_tensor = self.raw_data.center_label.reshape(self._n_batches, -1, 3).to(self.device)
        size_labels_tensor = self.raw_data.size_label.reshape(self._n_batches, -1, 3).to(self.device)
        sem_cls_label_tensor = self.raw_data.sem_cls_label.reshape(self._n_batches, -1).to(self.device)

        centre_labels, size_labels, class_labels = [], [], []
        for i in range(len(box_mask)):
            centre_labels.append(centre_labels_tensor[i][box_mask[i]])
            size_labels.append(size_labels_tensor[i][box_mask[i]])
            class_labels.append(sem_cls_label_tensor[i][box_mask[i]])
        return centre_labels, size_labels, class_labels

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss.backward()

    @staticmethod
    def anchors(size):
        anchors = [[1.0, 1.0, 1.0]]
        for a in [2.0, 4.0, 0.5, 0.25]:
            s = math.sqrt(a)
            anchors.append([s, s, 1.0 / s])
            anchors.append([1.0 / s, s, s])
            anchors.append([s, 1.0 / s, s])
        return size * torch.tensor(anchors).float()
