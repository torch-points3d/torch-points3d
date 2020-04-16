from typing import Dict
import torchnet as tnt
import torch
from src.models.base_model import BaseModel
from .base_tracker import BaseTracker
from .registration_metrics import compute_accuracy
from .registration_metrics import estimate_transfo
from .registration_metrics import get_matches
from .registration_metrics import fast_global_registration
from .registration_metrics import compute_hit_ratio
from .registration_metrics import compute_transfo_error


class PatchRegistrationTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """
        generic tracker for registration task.
        to track results, it measures the loss, and the accuracy.
        only useful for the training.
        """

        super(PatchRegistrationTracker, self).__init__(stage, wandb_log, use_tensorboard)

        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)

    def track(self, model: BaseModel):
        """ Add model predictions (accuracy)
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        N = len(outputs) // 2

        self._acc = compute_accuracy(outputs[:N], outputs[N:])

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        return metrics


class FragmentRegistrationTracker(BaseTracker):
    def __init__(
        self,
        dataset,
        num_points=5000,
        tau_1=0.1,
        tau_2=0.05,
        stage="train",
        wandb_log=False,
        use_tensorboard: bool = False,
    ):

        """
        tracker for registration tasks (we learn feature for each fragments like segmentation network)
it measures loss, feature match recall, hit ratio, rotation error, translation error.
        """
        super(FragmentRegistrationTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.reset(stage)
        self.num_points = num_points
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def reset(self, stage="train"):
        super().reset(stage=stage)

    def track(self, model: BaseModel):

        xyz = model.xyz
        xyz_target = model.xyz_target
        matches_gt = torch.stack([model.ind, model.ind_target]).T
        feat = model.outputs
        feat_target = model.outputs_target

        T_gt = estimate_transfo(xyz, xyz_target, matches_gt)
        match_pred = get_matches(feat, feat_target, num_matches=self.num_matches)
        T_pred = fast_global_registration(xyz, xyz_target, match_pred)

        self._hit_ratio = compute_hit_ratio(xyz, xyz_target, match_pred, T_gt, self.tau_1).item()
        trans_error, rot_error = compute_transfo_error(T_pred, T_gt)
        self._trans_error = trans_error.item()
        self._rot_error = rot_error.item()

    def get_metrics(self, verbose=False):
        metrics = super().get_metrics(verbose)
        if self.stage != "train":
            metrics["{}_hit_ratio".format(self._stage)] = self._hit_ratio
            # metrics["{}_feat_match_ratio".format(self._stage)] = self._feat_match_ratio
            metrics["{}_trans_error".format(self._stage)] = self._trans_error
            metrics["{}_rot_error".format(self._stage)] = self._rot_error
        return metrics
