import os
import copy
import hydra
from omegaconf import OmegaConf

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, convert_to_lightning_data_module
from torch_points3d.models.model_factory import instantiate_model, convert_to_lightning_module

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset
from pytorch_lightning import Trainer


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    dataset: BaseDataset = instantiate_dataset(cfg.data)
    model: BaseModel = instantiate_model(copy.deepcopy(cfg), dataset)
    model.instantiate_optimizers(cfg)
    model.set_pretrained_weights()

    # Set dataloaders
    dataset.create_dataloaders(
        model,
        cfg.training.batch_size,
        cfg.training.shuffle,
        cfg.training.num_workers,
        model.conv_type == "PARTIAL_DENSE" and getattr(cfg.training, "precompute_multi_scale", False),
    )
    
    # Verify attributes in dataset
    model.verify_data(dataset.train_dataset[0])

    data_module = convert_to_lightning_data_module(dataset)
    model = convert_to_lightning_module(model)
    model.tracker_options = cfg.get("tracker_options", {})
    model.trackers = data_module.trackers

    trainer = Trainer(
        max_epochs=2, 
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=False)
    
    trainer.fit(model, data_module)
    trainer.test()

    # https://github.com/facebookresearch/hydra/issues/440
    hydra._internal.hydra.GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
