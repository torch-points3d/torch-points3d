import torch
from typing import List
from torch_points_kernels import instance_iou


def offset_loss(pred_offsets, gt_offsets, total_instance_points):
    """ Computes the L1 norm between prediction and ground truth and
    also computes cosine similarity between both vectors.
    see https://arxiv.org/pdf/2004.01658.pdf equations 2 and 3
    """
    pt_diff = pred_offsets - gt_offsets
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
    offset_norm_loss = torch.sum(pt_dist) / (total_instance_points + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pred_offsets_norm = torch.norm(pred_offsets, p=2, dim=1)
    pred_offsets_ = pred_offsets / (pred_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = -(gt_offsets_ * pred_offsets_).sum(-1)  # (N)
    offset_dir_loss = torch.sum(direction_diff) / (total_instance_points + 1e-6)

    return {"offset_norm_loss": offset_norm_loss, "offset_dir_loss": offset_dir_loss}


def instance_iou_loss(
    predicted_clusters: List[torch.Tensor],
    cluster_scores: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    min_iou_threshold=0.25,
    max_iou_threshold=0.75,
):
    """ Loss that promotes higher scores for clusters with higher instance iou,
    see https://arxiv.org/pdf/2004.01658.pdf equation (7)
    """
    assert len(predicted_clusters) == cluster_scores.shape[0]
    ious = instance_iou(predicted_clusters, instance_labels, batch).max(1)[0]
    lower_mask = ious < min_iou_threshold
    higher_mask = ious > max_iou_threshold
    middle_mask = torch.logical_and(torch.logical_not(lower_mask), torch.logical_not(higher_mask))
    assert torch.sum(lower_mask + higher_mask + middle_mask) == ious.shape[0]
    shat = torch.zeros_like(ious)
    iou_middle = ious[middle_mask]
    shat[higher_mask] = 1
    shat[middle_mask] = (iou_middle - min_iou_threshold) / (max_iou_threshold - min_iou_threshold)
    return torch.nn.functional.binary_cross_entropy(cluster_scores, shat)
