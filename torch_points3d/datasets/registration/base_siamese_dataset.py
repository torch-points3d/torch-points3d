from torch_points3d.core.data_transform import MultiScaleTransform
from torch_points3d.core.data_transform import PairTransform
from torch_points3d.datasets.registration.pair import DensePairBatch
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.datasets.registration.pair import PairMultiScaleBatch, PairBatch
from torch_points3d.datasets.base_dataset import BaseDataset



class BaseSiameseDataset(BaseDataset):
    def __init__(self, dataset_opt):
        """
        base dataset for siamese inputs
        """
        super().__init__(dataset_opt)

    @staticmethod
    def _get_collate_function(conv_type, is_multiscale):

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)

        if is_multiscale:
            if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower():
                return lambda datalist: PairMultiScaleBatch.from_data_list(datalist)
            else:
                raise NotImplementedError(
                    "MultiscaleTransform is activated and supported only for partial_dense format"
                )

        if is_dense:
            return lambda datalist: DensePairBatch.from_data_list(datalist)
        else:
            return lambda datalist: PairBatch.from_data_list(datalist)
