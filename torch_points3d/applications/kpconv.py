from . import ModelFactory

from torch_points3d.core.data_transform import AddOnes


class KPConv(ModelFactory):

    MODULE_NAME = "KPConv"

    _transforms = [AddOnes()]
    _list_add_to_x = [True]
    _feat_names = ["ones"]
    _input_nc_feats = [1]
    _delete_feats = [True]

    def __init__(self, *args, **kwargs):
        super(KPConv, self).__init__(*args, **kwargs)

        self._build()

    @property
    def num_features(self):
        return len(self._feat_names)
