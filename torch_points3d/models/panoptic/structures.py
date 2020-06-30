import torch
from typing import NamedTuple, List


class PanopticResults(NamedTuple):
    semantic_logits: torch.Tensor
    offset_logits: torch.Tensor
    cluster_scores: torch.Tensor  # One float value per cluster
    clusters: List[torch.Tensor]  # Each item contains the list of indices in the cluster
    cluster_type: torch.Tensor  # Wether a cluster is coming from the votes or the original points. 0->original pos, 1->vote


class PanopticLabels(NamedTuple):
    center_label: torch.Tensor
    y: torch.Tensor
    num_instances: torch.Tensor
    instance_labels: torch.Tensor
    instance_mask: torch.Tensor
    vote_label: torch.Tensor
