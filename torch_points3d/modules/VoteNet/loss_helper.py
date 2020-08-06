""" Adapted from VoteNet

Ref: https://github.com/facebookresearch/votenet/blob/master/models/loss_helper.py
"""
import torch
import torch.nn as nn
import numpy as np

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point TODO should not be hardcoded
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

from torch_points3d.core.losses import huber_loss, nn_distance
from .votenet_results import VoteNetResults


def compute_vote_loss(input, output: VoteNetResults):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = output["seed_pos"].shape[0]
    num_seed = output["seed_pos"].shape[1]  # B,num_seed,3
    vote_xyz = output["seed_votes"]  # B,num_seed*vote_factor,3
    seed_inds = output["seed_inds"].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    if seed_inds.dim() == 1:
        seed_gt_votes_mask = torch.gather(input["vote_label_mask"], 0, seed_inds).view((batch_size, -1))
        seed_gt_votes = torch.gather(input["vote_label"], 0, seed_inds.unsqueeze(-1).repeat(1, 3 * GT_VOTE_FACTOR))
        seed_gt_votes += output["seed_pos"].view((-1, 3)).repeat((1, 3))
    else:
        seed_gt_votes_mask = torch.gather(input["vote_label_mask"], 1, seed_inds)
        seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
        seed_gt_votes = torch.gather(input["vote_label"], 1, seed_inds_expand)
        seed_gt_votes += output["seed_pos"].repeat(1, 1, 3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(
        batch_size * num_seed, -1, 3
    )  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(
        batch_size * num_seed, GT_VOTE_FACTOR, 3
    )  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
    return vote_loss


def compute_objectness_loss(inputs, outputs: VoteNetResults, loss_params):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_aggregated) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_aggregated) Tensor with value 0 or 1
        object_assignment: (batch_size, num_aggregated) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Compute objectness loss
    objectness_scores = outputs["objectness_scores"]
    weights = torch.tensor(loss_params.objectness_cls_weights).to(objectness_scores.device)
    criterion = nn.CrossEntropyLoss(weights, reduction="none")
    objectness_loss = criterion(objectness_scores.transpose(2, 1), outputs.objectness_label)
    objectness_loss = torch.sum(objectness_loss * outputs.objectness_mask) / (torch.sum(outputs.objectness_mask) + 1e-6)

    return objectness_loss


def compute_box_and_sem_cls_loss(inputs, outputs, loss_params):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = loss_params.num_heading_bin
    mean_size_arr = np.asarray(loss_params.mean_size_arr)
    num_size_cluster = len(mean_size_arr)

    object_assignment = outputs.object_assignment
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = outputs["center"]
    gt_center = inputs["gt_center"]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    box_label_mask = inputs["box_label_mask"]
    objectness_label = outputs["objectness_label"].float()
    centroid_reg_loss1 = torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
    centroid_reg_loss2 = torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(inputs["heading_class_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction="none")

    heading_class_loss = criterion_heading_class(
        outputs["heading_scores"].transpose(2, 1), heading_class_label.long()
    )  # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    heading_residual_label = torch.gather(
        inputs["heading_residual_label"], 1, object_assignment
    )  # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.zeros(batch_size, heading_class_label.shape[1], num_heading_bin).to(inputs.pos.device)
    heading_label_one_hot.scatter_(
        2, heading_class_label.unsqueeze(-1).long(), 1
    )  # src==1 so it's *one-hot* (B,K,num_heading_bin) TODO change that for pytorch OneHot
    heading_residual_normalized_loss = huber_loss(
        torch.sum(outputs["heading_residuals_normalized"] * heading_label_one_hot, -1)
        - heading_residual_normalized_label,
        delta=1.0,
    )  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (
        torch.sum(objectness_label) + 1e-6
    )

    # Compute size loss
    size_class_label = torch.gather(inputs["size_class_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction="none")
    if num_size_cluster != 0:
        size_class_loss = criterion_size_class(outputs["size_scores"].transpose(2, 1), size_class_label.long())  # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        size_residual_label = torch.gather(
            inputs["size_residual_label"], 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3)
        )  # select (B,K,3) from (B,K2,3)

        size_label_one_hot = torch.zeros(batch_size, size_class_label.shape[1], num_size_cluster).to(inputs.pos.device)
        size_label_one_hot.scatter_(
            2, size_class_label.unsqueeze(-1).long(), 1
        )  # src==1 so it's *one-hot* (B,K,num_size_cluster)
        size_label_one_hot_tiled = (
            size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3).contiguous()
        )  # (B,K,num_size_cluster,3)
        predicted_size_residual_normalized = torch.sum(
            outputs["size_residuals_normalized"].contiguous() * size_label_one_hot_tiled, 2
        )  # (B,K,3)

        mean_size_arr_expanded = (
            torch.from_numpy(mean_size_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(inputs.pos.device)
        )  # (1,1,num_size_cluster,3)
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
        size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
        size_residual_normalized_loss = torch.mean(
            huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1
        )  # (B,K,3) -> (B,K)
        size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label) / (
            torch.sum(objectness_label) + 1e-6
        )
    else:
        size_class_loss = 0
        size_residual_normalized_loss = 0

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(inputs["sem_cls_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction="none")
    sem_cls_loss = criterion_sem_cls(outputs["sem_cls_scores"].transpose(2, 1), sem_cls_label.long())  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return (
        center_loss,
        heading_class_loss,
        heading_residual_normalized_loss,
        size_class_loss,
        size_residual_normalized_loss,
        sem_cls_loss,
    )


def to_dense_labels(data):

    if data.batch is not None:
        batch_size = len(torch.unique(data.batch))
    else:
        batch_size = data.pos.shape[0]

    data["heading_class_label"] = data["heading_class_label"].view((batch_size, -1))
    data["heading_residual_label"] = data["heading_residual_label"].view((batch_size, -1))
    data["size_class_label"] = data["size_class_label"].view((batch_size, -1))
    data["size_residual_label"] = data["size_residual_label"].view((batch_size, -1, 3))
    data["sem_cls_label"] = data["sem_cls_label"].view((batch_size, -1))
    data["instance_box_corners"] = data["instance_box_corners"].view((batch_size, -1, 8, 3))
    data["box_label_mask"] = data["box_label_mask"].view((batch_size, -1))
    if data["center_label"].dim() == 3:
        data["gt_center"] = data["center_label"][:, :, 0:3]
    else:
        data["gt_center"] = data["center_label"][:, 0:3].view((batch_size, -1, 3))
    return data


def get_loss(inputs, outputs: VoteNetResults, loss_params):
    losses = {}

    inputs = to_dense_labels(inputs)

    # Vote loss
    vote_loss = compute_vote_loss(inputs, outputs)
    losses["vote_loss"] = vote_loss

    # Obj loss
    objectness_loss = compute_objectness_loss(inputs, outputs, loss_params)
    losses["objectness_loss"] = objectness_loss

    # Box loss and sem cls loss
    (
        center_loss,
        heading_cls_loss,
        heading_reg_loss,
        size_cls_loss,
        size_reg_loss,
        sem_cls_loss,
    ) = compute_box_and_sem_cls_loss(inputs, outputs, loss_params)
    losses["center_loss"] = center_loss
    losses["heading_cls_loss"] = heading_cls_loss
    losses["heading_reg_loss"] = heading_reg_loss
    losses["size_cls_loss"] = size_cls_loss
    losses["size_reg_loss"] = size_reg_loss
    losses["sem_cls_loss"] = sem_cls_loss
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
    losses["box_loss"] = box_loss

    # Final loss function
    loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
    loss *= 10  # TODO WHY???
    losses["loss"] = loss

    return losses
