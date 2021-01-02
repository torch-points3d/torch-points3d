import os
import copy
import hydra
from omegaconf import OmegaConf
import torch

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, convert_to_lightning_data_module
from torch_points3d.models.model_factory import instantiate_model, convert_to_lightning_module
from hydra.utils import instantiate
# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset
from pytorch_lightning import Trainer, _logger as log, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    seed_everything(42)

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

    monitor = getattr(cfg, "monitor", None)
    callbacks = []
    if monitor is not None:
        log.info(os.getcwd())
        callbacks = [
            ModelCheckpoint(
                monitor=monitor, 
                save_top_k=-1,
                filename='{epoch}-{'+f'{monitor}'+':.2f}',
                mode="max", 
            ),
            EarlyStopping(monitor=monitor)
        ]

    trainer = Trainer(**cfg.trainer, callbacks=callbacks)
    
    trainer.fit(model, data_module)
    
    if monitor is None:
        trainer.test(model)
    else:
        # Bug to resolve on Pytorch Side
        is_dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist_initialized:
            best_model_path = trainer.checkpoint_callback.best_model_path
            trainer.checkpoint_callback.best_model_path = trainer.accelerator_backend.broadcast(best_model_path)
        trainer.test()

    # https://github.com/facebookresearch/hydra/issues/440
    hydra._internal.hydra.GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
