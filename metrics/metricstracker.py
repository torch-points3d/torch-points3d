import numpy as np
import torchnet as tnt
import torch
from metrics.confusionmatrix import ConfusionMatrix


def meter_value(meter, dim=0):
    return meter.value()[dim] if meter.n > 0 else 0


class SegmentationTracker:

    def __init__(self, num_classes, stage="train", tensorboard_dir=None):
        """ Use the tracker to track an epoch. You can use the reset function before you start a new epoch

        Arguments:
            num_classes  -- number of classes

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            tensorboard_dir {str} -- Directory for tensorboard logging
        """
        self._num_classes = num_classes
        self._stage = stage
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            print("Show tensorboard metrics with the command <tensorboard --logdir={}>".format(tensorboard_dir))
            self._writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            self._writer = None
        self._n_iter = 0
        self.reset(stage)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def reset(self, stage="train"):
        self._stage = stage

        self._loss_meter = tnt.meter.AverageValueMeter()
        self._acc_meter = tnt.meter.AverageValueMeter()
        self._macc_meter = tnt.meter.AverageValueMeter()
        self._miou_meter = tnt.meter.AverageValueMeter()
        self._confusion_matrix = ConfusionMatrix(self._num_classes)

    @staticmethod
    def _convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    def track(self, loss, outputs, targets):
        """ Add current model predictions (usually the result of a batch) to the tracking

        Arguments:
            loss -- main loss
            outputs -- model predictions (NxK) where K is the number of labels
            targets -- target values NxK
        """
        self._loss_meter.add(loss.item())
        outputs = self._convert(outputs)
        targets = self._convert(targets)

        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        confusion_matrix_tmp = ConfusionMatrix(self._num_classes)
        confusion_matrix_tmp.count_predicted_batch(targets, np.argmax(outputs, 1))
        self._acc_meter.add(100 * confusion_matrix_tmp.get_overall_accuracy())
        self._macc_meter.add(100 * confusion_matrix_tmp.get_mean_class_accuracy())
        self._miou_meter.add(100 * confusion_matrix_tmp.get_average_intersection_union())

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
