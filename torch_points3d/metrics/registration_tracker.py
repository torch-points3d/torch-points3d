from typing import Dict, Any
import torchnet as tnt
import torch

from .base_tracker import BaseTracker
from .registration_metrics import compute_accuracy
from .registration_metrics import compute_hit_ratio
from .registration_metrics import compute_transfo_error
from torch_points3d.models import model_interface
from torch_points3d.utils.registration import estimate_transfo, fast_global_registration, get_matches


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

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add model predictions (accuracy)
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        N = len(outputs) // 2

        self._acc = compute_accuracy(outputs[:N], outputs[N:])

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {"loss": min, "acc": max}
        return self._metric_func


class FragmentRegistrationTracker(BaseTracker):
    def __init__(
        self,
        dataset,
        num_points=5000,
        tau_1=0.1,
        tau_2=0.05,
        rot_thresh=5,
        trans_thresh=2,
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
        self.rot_thresh = rot_thresh
        self.trans_thresh = trans_thresh

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._rot_error = tnt.meter.AverageValueMeter()
        self._trans_error = tnt.meter.AverageValueMeter()
        self._hit_ratio = tnt.meter.AverageValueMeter()
        self._feat_match_ratio = tnt.meter.AverageValueMeter()
        self._rre = tnt.meter.AverageValueMeter()
        self._rte = tnt.meter.AverageValueMeter()

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        super().track(model)
        if self._stage != "train":
            batch_idx, batch_idx_target = model.get_batch()
            # batch_xyz, batch_xyz_target = model.get_xyz()  # type: ignore
            # batch_ind, batch_ind_target, batch_size_ind = model.get_ind()  # type: ignore
            input, input_target = model.get_input()
            batch_xyz, batch_xyz_target = input.pos, input_target.pos
            batch_ind, batch_ind_target, batch_size_ind = input.ind, input_target.ind, input.size
            batch_feat, batch_feat_target = model.get_output()

            nb_batches = batch_idx.max() + 1
            cum_sum = 0
            cum_sum_target = 0
            begin = 0
            end = batch_size_ind[0].item()
            for b in range(nb_batches):
                xyz = batch_xyz[batch_idx == b]
                xyz_target = batch_xyz_target[batch_idx_target == b]
                feat = batch_feat[batch_idx == b]
                feat_target = batch_feat_target[batch_idx_target == b]
                # as we have concatenated ind,
                # we need to substract the cum_sum because we deal
                # with each batch independently
                # ind = batch_ind[b * len(batch_ind) / nb_batches : (b + 1) * len(batch_ind) / nb_batches] - cum_sum
                # ind_target = (batch_ind_target[b * len(batch_ind_target) / nb_batches : (b + 1) * len(batch_ind_target) / nb_batches]- cum_sum_target)
                ind = batch_ind[begin:end] - cum_sum
                ind_target = batch_ind_target[begin:end] - cum_sum_target
                # print(begin, end)
                if b < nb_batches - 1:
                    begin = end
                    end = begin + batch_size_ind[b + 1].item()
                cum_sum += len(xyz)
                cum_sum_target += len(xyz_target)
                rand = torch.randperm(len(feat))[: self.num_points]
                rand_target = torch.randperm(len(feat_target))[: self.num_points]

                matches_gt = torch.stack([ind, ind_target]).transpose(0, 1)

                # print(matches_gt.max(0), len(xyz), len(xyz_target), len(matches_gt))
                # print(batch_ind.shape, nb_batches)
                T_gt = estimate_transfo(xyz[matches_gt[:, 0]], xyz_target[matches_gt[:, 1]])

                matches_pred = get_matches(feat[rand], feat_target[rand_target])
                T_pred = fast_global_registration(
                    xyz[rand][matches_pred[:, 0]], xyz_target[rand_target][matches_pred[:, 1]]
                )

                hit_ratio = compute_hit_ratio(
                    xyz[rand][matches_pred[:, 0]], xyz_target[rand_target][matches_pred[:, 1]], T_gt, self.tau_1
                )

                trans_error, rot_error = compute_transfo_error(T_pred, T_gt)
                self._hit_ratio.add(hit_ratio.item())
                self._feat_match_ratio.add(float(hit_ratio.item() > self.tau_2))
                self._trans_error.add(trans_error.item())
                self._rot_error.add(rot_error.item())
                self._rre.add(rot_error.item() < self.rot_thresh)
                self._rte.add(trans_error.item() < self.trans_thresh)

    def get_metrics(self, verbose=False):
        metrics = super().get_metrics(verbose)
        if self._stage != "train":
            metrics["{}_hit_ratio".format(self._stage)] = float(self._hit_ratio.value()[0])
            metrics["{}_feat_match_ratio".format(self._stage)] = float(self._feat_match_ratio.value()[0])
            metrics["{}_trans_error".format(self._stage)] = float(self._trans_error.value()[0])
            metrics["{}_rot_error".format(self._stage)] = float(self._rot_error.value()[0])
            metrics["{}_rre".format(self._stage)] = float(self._rre.value()[0])
            metrics["{}_rte".format(self._stage)] = float(self._rte.value()[0])
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {
            "loss": min,
            "hit_ratio": max,
            "feat_match_ratio": max,
            "trans_error": min,
            "rot_error": min,
            "rre": max,
            "rte": max,
        }
        return self._metric_func
