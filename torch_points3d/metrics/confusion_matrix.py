import numpy as np
import torch
import os

import pytorch_lightning.metrics as plm





class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions.
    Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, number_of_labels=2):
        self.number_of_labels = number_of_labels
        self.confusion_matrix = None

    @staticmethod
    def create_from_matrix(confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        matrix = ConfusionMatrix(confusion_matrix.shape[0])
        matrix.confusion_matrix = confusion_matrix
        return matrix

    def count_predicted_batch(self, ground_truth_vec, predicted):
        assert np.max(predicted) < self.number_of_labels
        if torch.is_tensor(ground_truth_vec):
            ground_truth_vec = ground_truth_vec.numpy()
        if torch.is_tensor(predicted):
            predicted = predicted.numpy()
        batch_confusion = np.bincount(
            self.number_of_labels * ground_truth_vec.astype(int) + predicted, minlength=self.number_of_labels ** 2
        ).reshape(self.number_of_labels, self.number_of_labels)
        if self.confusion_matrix is None:
            self.confusion_matrix = batch_confusion
        else:
            self.confusion_matrix += batch_confusion

    def get_count(self, ground_truth, predicted):
        """labels are integers from 0 to number_of_labels-1"""
        return self.confusion_matrix[ground_truth][predicted]

    def get_confusion_matrix(self):
        """returns list of lists of integers; use it as result[ground_truth][predicted]
            to know how many samples of class ground_truth were reported as class predicted"""
        return self.confusion_matrix

    def get_intersection_union_per_class(self):
        """ Computes the intersection over union of each class in the
        confusion matrix
        Return:
            (iou, missing_class_mask) - iou for class as well as a mask highlighting existing classes
        """
        TP_plus_FN = np.sum(self.confusion_matrix, axis=0)
        TP_plus_FP = np.sum(self.confusion_matrix, axis=1)
        TP = np.diagonal(self.confusion_matrix)
        union = TP_plus_FN + TP_plus_FP - TP
        iou = 1e-8 + TP / (union + 1e-8)
        existing_class_mask = union > 1e-3
        return iou, existing_class_mask

    def get_overall_accuracy(self):
        """returns 64-bit float"""
        confusion_matrix = self.confusion_matrix
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                all_values += confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def get_average_intersection_union(self, missing_as_one=False):
        """ Get the mIoU metric by ignoring missing labels.
        If missing_as_one is True then treats missing classes in the IoU as 1
        """
        values, existing_classes_mask = self.get_intersection_union_per_class()
        if np.sum(existing_classes_mask) == 0:
            return 0
        if missing_as_one:
            values[~existing_classes_mask] = 1
            existing_classes_mask[:] = True
        return np.sum(values[existing_classes_mask]) / np.sum(existing_classes_mask)

    def get_mean_class_accuracy(self):  # added
        re = 0
        label_presents = 0
        for i in range(self.number_of_labels):
            total_gt = np.sum(self.confusion_matrix[i, :])
            if total_gt:
                label_presents += 1
                re = re + self.confusion_matrix[i][i] / max(1, total_gt)
        if label_presents == 0:
            return 0
        return re / label_presents

    def count_gt(self, ground_truth):
        return self.confusion_matrix[ground_truth, :].sum()


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=5)

    template_path = os.path.join(path2save, "{}.svg")
    # PRECISION
    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    # RECALL
    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")





def compute_overall_accuracy(confusion_matrix: torch.Tensor):
    """
    compute overall accuracy from confusion matrix
    Parameters

    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    """
    all_values = confusion_matrix.sum()
    matrix_diagonal = torch.trace(confusion_matrix)
    if all_values == 0:
        return matrix_diagonal
    else:
        return matrix_diagonal.float() / all_values


def compute_intersection_union_per_class(confusion_matrix: torch.Tensor, return_existing_mask=False):
    """
    compute intersection over unionper class from confusion matrix
    Parameters

    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    """

    TP_plus_FN = confusion_matrix.sum(0)
    TP_plus_FP = confusion_matrix.sum(1)
    TP = torch.diagonal(confusion_matrix)
    union = TP_plus_FN + TP_plus_FP - TP
    iou = 1e-8 + TP / (union + 1e-8)
    existing_class_mask = union > 1e-3
    if return_existing_mask:
        return iou, existing_class_mask
    else:
        return iou

def compute_average_intersection_union(
        confusion_matrix: torch.Tensor, missing_as_one: bool=False):
    """
    compute intersection over union on average from confusion matrix
    Parameters

    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    missing_as_one: bool, default: False
    """

    values, existing_classes_mask = compute_intersection_union_per_class(confusion_matrix, return_existing_mask=True)
    if torch.sum(existing_classes_mask) == 0:
        return torch.sum(existing_classes_mask)
    if missing_as_one:
        values[~existing_classes_mask] = 1
        existing_classes_mask[:] = True
    return torch.sum(values[existing_classes_mask]) / torch.sum(existing_classes_mask)


def compute_mean_class_accuracy(confusion_matrix: torch.Tensor):
    """
    compute intersection over union on average from confusion matrix
    Parameters

    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    """
    total_gts = confusion_matrix.sum(1)
    labels_presents = torch.where(total_gts > 0)[0]
    if(len(labels_presents) == 0):
        return total_gts[0]
    ones = torch.ones_like(total_gts)
    max_ones_total_gts = torch.cat([total_gts[None, :], ones[None, :]], 0).max(0)[0]
    re = (torch.diagonal(confusion_matrix)[labels_presents].float() / max_ones_total_gts[labels_presents]).sum()
    return re / float(len(labels_presents))
