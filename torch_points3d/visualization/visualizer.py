import os
import torch
import numpy as np
from matplotlib.cm import get_cmap
from math import log10, ceil
from plyfile import PlyData, PlyElement
import logging

log = logging.getLogger(__name__)


class Visualizer(object):
    """Initialize the Visualizer class.
    Parameters:
        viz_conf (OmegaConf Dictionnary) -- stores all config for the visualizer
        num_batches (dict) -- This dictionnary maps stage_name to #batches
        batch_size (int) -- Current batch size usef
        save_dir (str) -- The path used by hydra to store the experiment

    This class is responsible to save visual into .ply format and tensorboard mesh
    The configuration looks like that:
        visualization:
            activate: False # Wheter to activate the visualizer
            format: ["pointcloud", "tensorboard"] # image will come later
            num_samples_per_epoch: 2 # If negative, it will save all elements
            deterministic: True # False -> Randomly sample elements from epoch to epoch
            saved_keys: # Mapping from Data Object to structured numpy
                pos: [['x', 'float'], ['y', 'float'], ['z', 'float']]
                y: [['l', 'float']]
                pred: [['p', 'float']]
            indices: # List of indices to be saved (support "train", "test", "val")
                train: [0, 3]
            ply_format: binary_big_endian # PLY format (support "binary_big_endian", "binary_little_endian", "ascii")
            tensorboard_mesh: # Mapping from mesh name and propety use to color
                label: 'y'
                prediction: 'pred'

    """

    def __init__(self, viz_conf, num_batches, batch_size, save_dir, tracker):
        # From configuration and dataset
        for stage_name, stage_num_sample in num_batches.items():
            setattr(self, "{}_num_batches".format(stage_name), stage_num_sample)
        self._batch_size = batch_size
        self._activate = viz_conf.activate
        self._format = viz_conf.format
        self._num_samples_per_epoch = int(viz_conf.num_samples_per_epoch)
        self._deterministic = viz_conf.deterministic

        self._saved_keys = {}
        self._tensorboard_mesh = {}

        # Internal state
        self._stage = None
        self._current_epoch = None

        if "pointcloud" in self._format:
            self._saved_keys = viz_conf.saved_keys
            self._ply_format = viz_conf.ply_format if viz_conf.ply_format is not None else "binary_big_endian"

            # Current experiment path
            self._viz_path = os.path.join(save_dir, "viz")
            if not os.path.exists(self._viz_path):
                os.makedirs(self._viz_path)

        if "tensorboard" in self._format:
            if tracker._use_tensorboard:
                self._tensorboard_mesh = viz_conf.tensorboard_mesh

                # SummaryWriter for tensorboard loging
                self._writer = tracker._writer

        self._indices = {}
        self._contains_indices = False

        try:
            indices = getattr(viz_conf, "indices", None)
        except:
            indices = None

        if indices:
            for split in ["train", "test", "val"]:
                if split in indices:
                    split_indices = indices[split]
                    self._indices[split] = np.asarray(split_indices)
                    self._contains_indices = True

    def get_indices(self, stage):
        """This function is responsible to calculate the indices to be saved"""
        if self._contains_indices:
            return
        stage_num_batches = getattr(self, "{}_num_batches".format(stage))
        total_items = (stage_num_batches - 1) * self._batch_size
        if stage_num_batches > 0:
            if self._num_samples_per_epoch < 0:  # All elements should be saved.
                if stage_num_batches > 0:
                    self._indices[stage] = np.arange(total_items)
                else:
                    self._indices[stage] = None
            else:
                if self._deterministic:
                    if stage not in self._indices:
                        if self._num_samples_per_epoch > total_items:
                            log.warn("Number of samples to save is higher than the number of available elements")
                        self._indices[stage] = np.random.permutation(total_items)[: self._num_samples_per_epoch]
                else:
                    if self._num_samples_per_epoch > total_items:
                        log.warn("Number of samples to save is higher than the number of available elements")
                    self._indices[stage] = np.random.permutation(total_items)[: self._num_samples_per_epoch]

    @property
    def is_active(self):
        return self._activate

    def reset(self, epoch, stage):
        """This function is responsible to restore the visualizer
            to start a new epoch on a new stage
        """
        self._current_epoch = epoch
        self._seen_batch = 0
        self._stage = stage
        if self._activate:
            self.get_indices(stage)

    def _extract_from_PYG(self, item, pos_idx):
        num_samples = item.batch.shape[0]
        batch_mask = item.batch == pos_idx
        out_data = {}
        for k in item.keys:
            if torch.is_tensor(item[k]) and (k in self._saved_keys.keys() or k in self._tensorboard_mesh.values()):
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][batch_mask]
        return out_data

    def _extract_from_dense(self, item, pos_idx):
        assert (
            item.y.shape[0] == item.pos.shape[0]
        ), "y and pos should have the same number of samples. Something is probably wrong with your data to visualise"
        num_samples = item.y.shape[0]
        out_data = {}
        for k in item.keys:
            if torch.is_tensor(item[k]) and (k in self._saved_keys.keys() or k in self._tensorboard_mesh.values()):
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
        """This function is responsible to save the data into .ply objects
            Parameters:
                visuals (Dict[Data(pos=torch.Tensor, ...)]) -- Contains a dictionnary of tensors
            Make sure the saved_keys  within the config maps to the Data attributes.
        """
        if self._stage in self._indices:
            stage_num_batches = getattr(self, "{}_num_batches".format(self._stage))
            batch_indices = self._indices[self._stage] // self._batch_size
            pos_indices = self._indices[self._stage] % self._batch_size
            for idx in np.argwhere(self._seen_batch == batch_indices).flatten():
                pos_idx = pos_indices[idx]
                for visual_name, item in visuals.items():
                    if hasattr(item, "batch") and item.batch is not None:  # The PYG dataloader has been used
                        out_item = self._extract_from_PYG(item, pos_idx)
                    else:
                        out_item = self._extract_from_dense(item, pos_idx)

                    if hasattr(self, "_viz_path"):
                        dir_path = os.path.join(self._viz_path, str(self._current_epoch), self._stage)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)

                        filename = "{}_{}.ply".format(self._seen_batch, pos_idx)
                        path_out = os.path.join(dir_path, filename)

                        npy_array = self._dict_to_structured_npy(out_item)
                        el = PlyElement.describe(npy_array, visual_name)
                        if self._ply_format == "ascii":
                            PlyData([el], text=True).write(path_out)
                        elif self._ply_format == "binary_little_endian":
                            PlyData([el], byte_order="<").write(path_out)
                        elif self._ply_format == "binary_big_endian":
                            PlyData([el], byte_order=">").write(path_out)
                        else:
                            PlyData([el]).write(path_out)

                    if hasattr(self, "_writer"):
                        pos = out_item['pos'].detach().cpu().unsqueeze(0)
                        colors = get_cmap('tab10')
                        config_dict = {
                            "material": {
                                "size": 0.3
                            }
                        }

                        for label, k in self._tensorboard_mesh.items():
                            value = out_item[k].detach().cpu()

                            if len(value.shape) == 2 and value.shape[1] == 3:
                                if value.min() >= 0 and value.max() <= 1:
                                    value = (255*value).type(torch.uint8).unsqueeze(0)
                                else:
                                    value = value.type(torch.uint8).unsqueeze(0)
                            elif len(value.shape) == 1 and value.shape[0] == 1:
                                value = np.tile((255*colors(value.numpy() % 10))[:,0:3].astype(np.uint8), (pos.shape[0],1)).reshape((1,-1,3))
                            elif len(value.shape) == 1 or value.shape[1] == 1:
                                value = (255*colors(value.numpy() % 10))[:,0:3].astype(np.uint8).reshape((1,-1,3))
                            else:
                                continue

                            self._writer.add_mesh(self._stage + "/" + visual_name + "/" + label, pos, colors=value, config_dict=config_dict, global_step=(self._current_epoch-1)*(10**ceil(log10(stage_num_batches+1)))+self._seen_batch)
                        
            self._seen_batch += 1
