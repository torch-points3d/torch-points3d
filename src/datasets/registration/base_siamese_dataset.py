from src.core.data_transform import MultiScaleTransform
from src.core.data_transform import PairTransform
from src.datasets.registration.pair import SimplePairBatch
from src.utils.enums import ConvolutionFormat
from src.utils.config import ConvolutionFormatFactory
from src.datasets.registration.pair import PairMultiScaleBatch, PairBatch
from src.datasets.base_dataset import BaseDataset


class BaseSiameseDataset(BaseDataset):
    def __init__(self, dataset_opt):
        """
        base dataset for siamese inputs
        """
        super().__init__(dataset_opt)

    def _get_collate_function(self, conv_type, is_multiscale):

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)

        if is_multiscale:
            if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower():
                return lambda datalist: PairMultiScaleBatch.from_data_list(datalist)
            else:
                raise NotImplementedError(
                    "MultiscaleTransform is activated and supported only for partial_dense format"
                )

        if is_dense:
            return lambda datalist: SimplePairBatch.from_data_list(datalist)
        else:
            return lambda datalist: PairBatch.from_data_list(datalist)

    def set_strategies(self, model):
        strategies = model.get_spatial_ops()
        transform = PairTransform(MultiScaleTransform(strategies))
        self._set_multiscale_transform(transform)
