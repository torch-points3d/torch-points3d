import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement


class Visualizer(object):
    def __init__(self, viz_conf, num_batches, batch_size, run_path):
        # From configuration and dataset
        for stage_name, stage_num_sample in num_batches.items():
            setattr(self, "{}_num_batches".format(stage_name), stage_num_sample)
        self._batch_size = batch_size
        self._activate = viz_conf.activate
        self._format = viz_conf.format
        self._num_samples_per_epoch = int(viz_conf.num_samples_per_epoch)
        self._deterministic = viz_conf.deterministic

        self._saved_keys = viz_conf.saved_keys

        # Internal state
        self._stage = None
        self._current_epoch = None

        # Current experiment path
        self._run_path = run_path
        self._viz_path = os.path.join(self._run_path, "viz")
        if not os.path.exists(self._viz_path):
            os.makedirs(self._viz_path)

        self._indices = {}
        self._contains_indices = False

        try:
            indices = getattr(viz_conf, "indices", None)
        except:
            indices = None

        if indices:
            for split in ["train", "test", "val"]:
                if hasattr(indices, split):
                    indices = getattr(indices, split)
                    self._indices[split] = np.asarray(indices)
                    self._contains_indices = True

    def get_indices(self, stage):
        """This function is responsible to calculate the indices to be saved"""
        if self._contains_indices:
            return
        stage_num_batches = getattr(self, "{}_num_batches".format(stage))
        if stage_num_batches > 0:
            if self._num_samples_per_epoch < 0:  # All elements should be saved.
                if stage_num_batches > 0:
                    self._indices[stage] = np.arange(stage_num_batches)
                else:
                    self._indices[stage] = None
            else:
                if self._deterministic:
                    if stage not in self._indices:
                        if self._num_samples_per_epoch > stage_num_batches:
                            raise Exception("Number of samples to save if higher than the number of available elements")
                        self._indices[stage] = np.random.permutation(stage_num_batches)[: self._num_samples_per_epoch]
                else:
                    if self._num_samples_per_epoch > stage_num_batches:
                        raise Exception("Number of samples to save if higher than the number of available elements")
                    self._indices[stage] = np.random.permutation(stage_num_batches)[: self._num_samples_per_epoch]

    @property
    def is_active(self):
        return self._activate

    def reset(self, epoch, stage):
        self._current_epoch = epoch
        self._seen_batch = 0
        self._stage = stage
        self.get_indices(stage)

    def _extract_from_PYG(self, item, pos_idx):
        num_samples = item.batch.shape[0]
        batch_mask = item.batch == pos_idx
        out_data = {}
        for k in item.keys:
            if torch.is_tensor(item[k]) and k in self._saved_keys.keys():
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][batch_mask]
        return out_data

    def _extract_from_dense(self, item, pos_idx):
        num_samples = item.y.shape[0]
        out_data = {}
        for k in item.keys:
            if torch.is_tensor(item[k]) and k in self._saved_keys.keys():
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][pos_idx]
        return out_data

    def _dict_to_structured_npy(self, item):
        item.keys()
        out = []
        dtypes = []
        for k, v in item.items():
            v_npy = v.detach().cpu().numpy()
            if len(v_npy.shape) == 1:
                v_npy = v_npy[..., np.newaxis]
            for dtype in self._saved_keys[k]:
                dtypes.append(dtype)
            out.append(v_npy)
        out = np.concatenate(out, axis=-1)
        dtypes = np.dtype([tuple(d) for d in dtypes])
        return np.asarray([tuple(o) for o in out], dtype=dtypes)

    def save_visuals(self, visuals):
        if self._stage in self._indices:
            self._indices[self._stage]
            batch_indices = self._indices[self._stage] // self._batch_size
            pos_indices = self._indices[self._stage] % self._batch_size
            for idx in np.argwhere(self._seen_batch == batch_indices).flatten():
                pos_idx = pos_indices[idx]
                for k, item in visuals.items():
                    if hasattr(item, "batch"):  # The PYG dataloader has been used
                        out_item = self._extract_from_PYG(item, pos_idx)
                    else:
                        out_item = self._extract_from_dense(item, pos_idx)
                    out_item = self._dict_to_structured_npy(out_item)

                    dir_path = os.path.join(self._viz_path, str(self._current_epoch), self._stage)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    filename = "{}_{}.ply".format(self._seen_batch, pos_idx)
                    path_out = os.path.join(dir_path, filename)
                    el = PlyElement.describe(out_item, "sample")
                    PlyData([el], byte_order=">").write(path_out)
            self._seen_batch += 1
