import numpy as np
import torchnet as tnt
import torch
from metrics.confusionmatrix import ConfusionMatrix


def meter_value(meter, dim=0):
    return meter.value()[dim] if meter.n > 0 else 0


class SegmentationTracker:

    def __init__(self, num_classes, stage="train", log_dir=None):
        self._num_classes = num_classes
        self._stage = stage
        # self._best_selectors = {"loss": min, "acc": max, "miou": max, "std": min}
        # self._best_metrics = {}
        # self._internal_losses = []
        self._with_in_n_per = 1
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            print("Show tensorboard metrics with the command <tensorboard --logdir={}>".format(log_dir))
            self._writer = SummaryWriter(log_dir=log_dir)
        else:
            self._writer = None
        self._n_iter = 0
        self.reset(stage)

    def reset(self, stage="train"):
        self._stage = stage

        self._loss_meter = tnt.meter.AverageValueMeter()
        self._acc_meter = tnt.meter.AverageValueMeter()
        self._macc_meter = tnt.meter.AverageValueMeter()
        self._miou_meter = tnt.meter.AverageValueMeter()
        self._confusion_matrix = ConfusionMatrix(self._num_classes)

    @staticmethod
    def convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    # def track_internal_losses(self, internal_losses):
    #     self._internal_losses = internal_losses.keys()
    #     for loss_name, loss_value in internal_losses.items():
    #         if not isinstance(loss_value, int):
    #             if hasattr(self, loss_name):
    #                 getattr(self, loss_name).add(loss_value.item())
    #             else:
    #                 averageValueMeter = tnt.meter.AverageValueMeter()
    #                 averageValueMeter.add(loss_value.item())
    #                 setattr(self, loss_name, averageValueMeter)

    def track(self, loss, outputs, targets):
        """ Add current model stats to the tracking

        Arguments:
            loss -- main loss
            outputs -- model predictions (NxK) where K is the number of labels
            targets -- target values NxK
        """
        self._loss_meter.add(loss.item())
        outputs = self.convert(outputs)
        targets = self.convert(targets)

        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        confusion_matrix_tmp = ConfusionMatrix(self._num_classes)
        confusion_matrix_tmp.count_predicted_batch(targets, np.argmax(outputs, 1))
        self._acc_meter.add(100 * confusion_matrix_tmp.get_overall_accuracy())
        self._macc_meter.add(100 * confusion_matrix_tmp.get_mean_class_accuracy())
        self._miou_meter.add(100 * confusion_matrix_tmp.get_average_intersection_union())

    # def from_stats(self, stats):
    #     print('LOADING FROM STATS')
    #     if len(stats) == 0:
    #         print('STATS IS EMPTY')
    #         return
    #     self.n_iter = len(stats)  # Restart the iter
    #     for key in stats[-1].keys():
    #         if 'best' in key:

    #             if self.is_legacy(key):
    #             splits = key.replace('iou', 'miou').split('_')
    #             splits.insert(1, 'val')
    #             new_key = '_'.join(splits)
    #             print(new_key, stats[-1][key])
    #             setattr(self, new_key, stats[-1][key])
    #             else:
    #             print(key, stats[-1][key])
    #             setattr(self, key, stats[-1][key])

    # def get_best_metrics(self):
    #     self._best_metrics

    # def _check_not_only_internal_losses(self, improved_metrics, stage):
    #     internal_keys = ['{}_{}'.format(stage, k) for k in list(self._internal_losses)]
    #     for key_name in improved_metrics.keys():
    #         if key_name not in internal_keys:
    #             return improved_metrics
    #     return {}

    # def _add_metrics(self, metrics, stage):
    #     improved_metrics = {}
    #     for metric_name, metric_score in metrics.items():
    #         for mn, func in self._best_selectors.items():
    #             if mn not in metric_name:
    #                 continue

    #             ref = "best_{}".format(metric_name)
    #             if ref not in self._best_metrics:
    #                 self._best_metrics[ref] = metric_score
    #                 improved_metrics[metric_name] = metric_score
    #             else:
    #                 old_score = self._best_metrics[ref]
    #                 best_score = func(old_score, metric_score)
    #                 if self._with_in_n_per > 0:
    #                     ratio = 100 * np.abs(metric_score - old_score) / old_score
    #                     if ratio < self._with_in_n_per:
    #                         improved_metrics[metric_name] = metric_score
    #                 if metric_score == best_score:
    #                     improved_metrics[metric_name] = metric_score
    #                     self._best_metrics[ref] = best_score
    #     return self._check_not_only_internal_losses(improved_metrics, stage)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def _publish_to_tensorboard(self, metrics):
        if self._stage == "train":
            self._n_iter += 1

        for metric_name, metric_value in metrics.items():
            metric_name = "{}/{}".format(metric_name.replace(self._stage+"_", ""), self._stage)
            self._writer.add_scalar(metric_name, metric_value, self._n_iter)

    def get_metrics(self):
        metrics = {}
        metrics['{}_loss'.format(self._stage)] = meter_value(self._loss_meter)
        metrics['{}_acc'.format(self._stage)] = meter_value(self._acc_meter, dim=0)
        metrics['{}_macc'.format(self._stage)] = meter_value(self._macc_meter, dim=0)
        metrics['{}_miou'.format(self._stage)] = meter_value(self._miou_meter, dim=0)

        # Estimate lower bound performance based on the variance
        metrics['{}_lb_acc'.format(self._stage)] = meter_value(self._acc_meter, dim=0) - \
            meter_value(self._acc_meter, dim=1)
        metrics['{}_lb_macc'.format(self._stage)] = meter_value(self._macc_meter, dim=0) - \
            meter_value(self._macc_meter, dim=1)
        metrics['{}_lb_miou'.format(self._stage)] = meter_value(self._miou_meter, dim=0) - \
            meter_value(self._miou_meter, dim=1)

        # for loss_name in self._internal_losses:
        #     metrics['{}_{}'.format(self._stage, loss_name)] = getattr(self, loss_name).value()[0]

        # improved_metrics = self._add_metrics(metrics, self._stage)
        if self._writer:
            self._publish_to_tensorboard(metrics)
        return metrics
