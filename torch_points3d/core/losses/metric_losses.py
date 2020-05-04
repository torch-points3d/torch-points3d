import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec


def pdist(A, B, dist_type="L2"):
    if dist_type == "L2":
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == "SquareL2":
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError("Not implemented")


class ContrastiveHardestNegativeLoss(nn.Module):
    r"""
    Compute contrastive loss between positive pairs and mine negative pairs which are not in the intersection of the two point clouds (taken from https://github.com/chrischoy/FCGF)
    Let :math:`(f_i, f^{+}_i)_{i=1 \dots N}` set of positive_pairs and :math:`(f_i, f^{-}_i)_{i=1 \dots M}` a set of negative pairs
    The loss is computed as:
    .. math::
        L = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N [d^{+}_{ij} - \lambda_+]_+ + \frac{1}{M} \sum_{i=1}^M [\lambda_{-} - d^{-}_i]_+

    where:
    .. math::
        d^{+}_{ij} = ||f_{i} - f^{+}_{j}||

    and
    .. math::
        d^{-}_{i} = \min_{j}(||f_{i} - f^{-}_{j}||)

    In this loss, we only mine the negatives
    Parameters
    ----------

    pos_thresh:
        positive threshold of the positive distance
    neg_thresh:
        negative threshold of the negative distance
    num_pos:
        number of positive pairs
    num_hn_samples:
        number of negative point we mine.
    """

    def __init__(self, pos_thresh, neg_thresh, num_pos=5192, num_hn_samples=2048):
        nn.Module.__init__(self)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.num_pos = num_pos
        self.num_hn_samples = num_hn_samples

    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs, thresh=None):
        """
        Generate negative pairs
        """
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, self.num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, self.num_hn_samples), replace=False)

        if N_pos_pairs > self.num_pos:
            pos_sel = np.random.choice(N_pos_pairs, self.num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = pdist(posF0, subF1, dist_type="L2")
        D10 = pdist(posF1, subF0, dist_type="L2")

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def forward(self, F0, F1, matches, xyz0=None, xyz1=None):

        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(F0, F1, matches.detach().cpu())

        return pos_loss + neg_loss


class BatchHardContrastiveLoss(nn.Module):
    """
        apply contrastive loss but mine the negative sample in the batch.
    apply a mask if the distance between negative pair is too close.
    Parameters
    ----------
    pos_thresh:
        positive threshold of the positive distance
    neg_thresh:
        negative threshold of the negative distance
    min_dist:
        minimum distance to be in the negative sample
    """

    def __init__(self, pos_thresh, neg_thresh, min_dist=0.15):
        nn.Module.__init__(self)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.min_dist = min_dist

    def forward(self, F0, F1, positive_pairs, xyz0=None, xyz1=None):

        posF0 = F0[positive_pairs[:, 0]]
        posF1 = F1[positive_pairs[:, 1]]

        subxyz0 = xyz0[positive_pairs[:, 0]]
        false_negative = pdist(subxyz0, subxyz0, dist_type="L2") > self.min_dist
        # dists = pdist(posF0, posF1, dist_type="L2").view(-1)
        furthest_pos, _ = (posF0 - posF1).pow(2).max(1)
        neg_loss = F.relu(self.neg_thresh - (posF0[0] - posF1[false_negative[0]]).pow(2).sum(1).min()).pow(2) / len(
            posF0
        )

        for i in range(1, len(posF0)):
            neg_loss += F.relu(self.neg_thresh - (posF0[i] - posF1[false_negative[i]]).pow(2).sum(1).min()).pow(
                2
            ) / len(posF0)

        pos_loss = F.relu(furthest_pos - self.pos_thresh).pow(2)
        # neg_loss = F.relu(self.neg_thresh - closest_neg)
        return pos_loss.mean() + neg_loss.mean()
