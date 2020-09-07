from abc import abstractmethod

from torch_points3d.models import model_interface


class SiameseTrackerInterface(model_interface.TrackerInterface):
    @abstractmethod
    def get_trans_gt(self):
        """
        return the transformation between the source and the target
        """
