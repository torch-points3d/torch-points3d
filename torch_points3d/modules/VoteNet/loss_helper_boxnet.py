# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
from nn_distance import nn_distance, huber_loss

sys.path.append(BASE_DIR)
from loss_helper import compute_box_and_sem_cls_loss

OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points["aggregated_vote_xyz"]
    gt_center = end_points["center_label"][:, :, 0:3]
    gt_center.shape[0]
    aggregated_vote_xyz.shape[1]
    gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # NOTE: Different from VoteNet, here we use seed label as objectness label.
    seed_inds = end_points["seed_inds"].long()  # B,num_seed in [0,num_points-1]
    seed_gt_votes_mask = torch.gather(end_points["vote_label_mask"], 1, seed_inds)
    end_points["seed_labels"] = seed_gt_votes_mask
    aggregated_vote_inds = end_points["aggregated_vote_inds"]
    objectness_label = torch.gather(
        end_points["seed_labels"], 1, aggregated_vote_inds.long()
    )  # select (B,K) from (B,1024)
    objectness_mask = torch.ones(
        (objectness_label.shape[0], objectness_label.shape[1])
    ).cuda()  # no ignore zone anymore

    # Compute objectness loss
    objectness_scores = end_points["objectness_scores"]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction="none")
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(end_points)
    end_points["objectness_loss"] = objectness_loss
    end_points["objectness_label"] = objectness_label
    end_points["objectness_mask"] = objectness_mask
    end_points["object_assignment"] = object_assignment
    total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
    end_points["pos_ratio"] = torch.sum(objectness_label.float().cuda()) / float(total_num_proposal)
    end_points["neg_ratio"] = torch.sum(objectness_mask.float()) / float(total_num_proposal) - end_points["pos_ratio"]

    # Box loss and sem cls loss
    (
        center_loss,
        heading_cls_loss,
        heading_reg_loss,
        size_cls_loss,
        size_reg_loss,
        sem_cls_loss,
    ) = compute_box_and_sem_cls_loss(end_points, config)
    end_points["center_loss"] = center_loss
    end_points["heading_cls_loss"] = heading_cls_loss
    end_points["heading_reg_loss"] = heading_reg_loss
    end_points["size_cls_loss"] = size_cls_loss
    end_points["size_reg_loss"] = size_reg_loss
    end_points["sem_cls_loss"] = sem_cls_loss
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
    end_points["box_loss"] = box_loss

    # Final loss function
    loss = 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
    loss *= 10
    end_points["loss"] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points["objectness_scores"], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / (
        torch.sum(objectness_mask) + 1e-6
    )
    end_points["obj_acc"] = obj_acc

    return loss, end_points
