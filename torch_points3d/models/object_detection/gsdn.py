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
from torch_points3d.modules.GSDN.gsdn_results import GSDNResult
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.losses import huber_loss


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
        self.boxes = []
        anchor_size = 1
        for i in range(len(self.up_modules)):
            data, box = self.up_modules[i](data, stack_down.pop())

            # Set box anchors and grid_size
            box.anchors = self.anchors(anchor_size).to(box.device)
            box.grid_size = self.raw_data.grid_size[0].to(box.device)

            # append to hierarchy of predictions
            self.boxes.append(box)

            # increase anchor_size
            anchor_size *= 2

        self.output = self.boxes
        self.coord_manager = data.coords_man

    def _compute_losses(self):
        self.loss = 0
        self.anchor_loss = 0
        self.sparsity_loss = 0
        self.sem_loss = 0
        self.regr_loss = 0

        with torch.no_grad():
            centre_labels, size_labels, class_labels = self._extract_gt()
            for i in range(1, len(self.boxes) + 1):
                box = self.boxes[-i]
                box.evaluate_labels(centre_labels, size_labels, class_labels)
                if i > 1:
                    box.set_sparsity(self.boxes[-i + 1], self.coord_manager)

        #  Losses
        for box in self.boxes:
            # Only enforce anchor objectness on positive and negative boxes
            mask = torch.logical_or(box.positive_mask, box.negative_mask)
            logits = box.objectness.reshape(-1)[mask]
            labels = box.positive_mask.float()[mask]
            weight = torch.sum(box.negative_mask).float() / torch.sum(box.positive_mask).float()
            anchor_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
            if not torch.isnan(anchor_loss):
                self.anchor_loss += anchor_loss

            # Sparsity on positive and negative boxes
            mask = torch.logical_or(box.sparsity_positive, box.sparsity_negative)
            logits = box.sparsity.reshape(-1)[mask]
            labels = box.sparsity_positive.float()[mask]
            weight = torch.sum(box.sparsity_negative).float() / torch.sum(box.sparsity_positive).float()
            sparsity_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
            if not torch.isnan(sparsity_loss):
                self.sparsity_loss += sparsity_loss

            # semantic on positives
            positive_class_logits = box.class_logits[box.positive_mask]
            positive_label = box.class_labels[box.positive_mask]
            sem_loss = torch.nn.functional.cross_entropy(
                positive_class_logits, positive_label, ignore_index=IGNORE_LABEL
            )
            if not torch.isnan(sem_loss):
                self.sem_loss += sem_loss

            # centre on positives
            centre_logits = box.centre_logits[box.positive_mask]
            centre_labels = box.get_rescaled_centre_labels()[box.positive_mask]
            regr_loss = huber_loss(centre_labels - centre_logits).mean()
            if not torch.isnan(regr_loss):
                self.regr_loss += regr_loss

            # size on positives
            size_logits = box.size_logits[box.positive_mask]
            size_labels = box.get_rescaled_size_labels()[box.positive_mask]
            regr_loss = huber_loss(size_labels - size_logits).mean()
            if not torch.isnan(regr_loss):
                self.regr_loss += regr_loss

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
        self._compute_losses()
        if torch.is_tensor(self.loss):
            self.loss.backward()

    @staticmethod
    def anchors(size):
        anchors = [[1.0, 1.0, 1.0]]
        for a in [2.0, 0.5]:
            s = math.sqrt(a)
            anchors.append([s, s, 1.0 / s])
            anchors.append([1.0 / s, s, s])
            anchors.append([s, 1.0 / s, s])
        return size * torch.tensor(anchors).float()
