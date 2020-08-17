import torch_points3d.datasets.segmentation.shapenet as ShapeNetBase
from torch_points3d.metrics.generation_tracker import GenerationTracker

class ShapeNet(ShapeNetBase.ShapeNet):
    @property
    def num_features(self):
        return 0
    
class ShapeNetDataset(ShapeNetBase.ShapeNetDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        try:
            self._category = dataset_opt.category
            is_test = dataset_opt.get("is_test", False)
        except KeyError:
            self._category = None

        self.train_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test,
        )

        self.val_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            is_test=is_test,
        )

        self.test_dataset = ShapeNet(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            is_test=is_test,
        )
        self._categories = self.train_dataset.categories

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return GenerationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)