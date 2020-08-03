import torch
from torch_geometric.data import Data

from torch_points3d.core.spatial_ops import BaseSampler


class RandomSamplerToDense(BaseSampler):
    """ If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
        This class samples randomly points either in "DENSE" or "PARTIAL_DENSE"
        format and output the sampled points in "DENSE" format
    """

    def sample(self, data, num_batches, conv_type):
        if conv_type == "DENSE":
            assert (
                self._num_to_sample <= data.pos.shape[1]
            ), "self._num_to_sample: {} should be smaller than num_pos: {}".format(
                self._num_to_sample, data.pos.shape[1]
            )
            idx = torch.randint(0, data.pos.shape[1], (data.pos.shape[0], self._num_to_sample,)).to(data.pos.device)
            data.pos = torch.gather(data.pos, 1, idx.unsqueeze(-1).repeat(1, 1, data.pos.shape[-1]))
            data.x = torch.gather(data.x, 2, idx.unsqueeze(1).repeat(1, data.x.shape[1], 1))
            return data, idx
        else:
            pos = []
            x = []
            idx_out = []
            num_points = 0
            for batch_idx in range(num_batches):
                batch_mask = data.batch == batch_idx
                pos_masked = data.pos[batch_mask]
                x_masked = data.x[batch_mask]
                assert (
                    self._num_to_sample <= pos_masked.shape[0]
                ), "self._num_to_sample: {} should be smaller than num_pos: {}".format(
                    self._num_to_sample, pos_masked.shape[0]
                )
                idx = torch.randint(0, pos_masked.shape[0], (self._num_to_sample,))
                pos.append(pos_masked[idx])
                x.append(x_masked[idx])
                idx_out.append(idx + num_points)
                num_points += pos_masked.shape[0]
            data.pos = torch.stack(pos)
            data.x = torch.stack(x).permute(0, 2, 1)
            return data, torch.cat(idx_out, dim=0)


class FPSSamplerToDense(BaseSampler):
    """ If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
        This class samples randomly points either in "DENSE" or "PARTIAL_DENSE"
        format and output the sampled points in "DENSE" format
    """

    def sample(self, data, num_batches, conv_type):
        from torch_geometric.nn import fps

        if conv_type == "DENSE":
            assert (
                self._num_to_sample <= data.pos.shape[1]
            ), "self._num_to_sample: {} should be smaller than num_pos: {}".format(
                self._num_to_sample, data.pos.shape[1]
            )
            batch = torch.zeros(data.pos.shape[1]).to(data.pos.device).long()
            idx_out = []
            pos = []
            x = []
            for i in range(num_batches):
                idx = fps(data.pos[i], batch, ratio=self._get_ratio_to_sample(data.pos.shape[1]))
                idx_out.append(idx)
            idx_out = torch.stack(idx_out)
            pos = torch.gather(data.pos, 1, idx_out.unsqueeze(-1).repeat(1, 1, data.pos.shape[-1]))
            x = torch.gather(data.x, 2, idx_out.unsqueeze(1).repeat(1, data.x.shape[1], 1))
            return Data(pos=pos, x=x), idx_out
        else:
            pos = []
            x = []
            idx_out = []
            num_points = 0
            for batch_idx in range(num_batches):
                batch_mask = data.batch == batch_idx
                pos_masked = data.pos[batch_mask]
                x_masked = data.x[batch_mask]
                if self._num_to_sample >= pos_masked.shape[0]:
                    pos.append(pos_masked)
                    x.append(x_masked)
                    idx = torch.arange(0, pos_masked.shape[0]).to(pos_masked.device)
                    idx_out.append(idx + num_points)
                    num_points += pos_masked.shape[0]
                else:
                    batch = torch.zeros(pos_masked.shape[0]).to(pos_masked).long()
                    idx = fps(pos_masked, batch, ratio=self._get_ratio_to_sample(pos_masked.shape[0]))
                    pos.append(pos_masked[idx])
                    x.append(x_masked[idx])
                    idx_out.append(idx + num_points)
                    num_points += pos_masked.shape[0]

            sample_size = min([len(idx) for idx in idx_out])
            for i, idx in enumerate(idx_out):
                if len(idx) == sample_size:
                    continue
                idx_out[i] = idx[:sample_size]
                pos[i] = pos[i][:sample_size]
                x[i] = x[i][:sample_size]

            data_out = Data(pos=torch.stack(pos), x=torch.stack(x).permute(0, 2, 1))
            return data_out, torch.cat(idx_out, dim=0)
