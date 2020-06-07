import torch
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_geometric.nn.unpool import knn_interpolate


class SegmentationFullResHelpers:
    def __init__(self, raw_data, num_classes, conv_type, class_seg_map=None):
        self._raw_data = raw_data
        self._num_pos = raw_data.pos.shape[0]
        self._votes = torch.zeros((self._num_pos, num_classes), dtype=torch.float)
        self._vote_counts = torch.zeros(self._num_pos, dtype=torch.float)
        self._full_res_preds = None
        self._conv_type = conv_type
        self._class_seg_map = class_seg_map

    @property
    def full_res_labels(self):
        return self._raw_data.y

    @property
    def full_res_preds(self):
        self._predict_full_res()
        if self._class_seg_map:
            return self._full_res_preds[:, self._class_seg_map].argmax(1) + self._class_seg_map[0]
        else:
            return self._full_res_preds.argmax(-1)

    def add_vote(self, data, output, batch_mask):
        """ Populates scores for the points in data

        Parameters
        ----------
        data : Data
            should contain `pos` and `SaveOriginalPosId.KEY` keys
        output : torch.Tensor
            probablities out of the model, shape: [N,nb_classes]
        batch_mask: torch.Tensor | int
            mask to access the associated element
        """
        idx = data[SaveOriginalPosId.KEY][batch_mask]
        self._votes[idx] += output
        self._vote_counts[idx] += 1

    def _predict_full_res(self):
        """ Predict full resolution results based on votes """
        has_prediction = self._vote_counts > 0
        votes = self._votes[has_prediction].div(self._vote_counts[has_prediction].unsqueeze(-1))

        # Upsample and predict
        full_pred = knn_interpolate(votes, self._raw_data.pos[has_prediction], self._raw_data.pos, k=1)
        self._full_res_preds = full_pred

    def __repr__(self):
        return "{}(num_pos={})".format(self.__class__.__name__, self._num_pos)
