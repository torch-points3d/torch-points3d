import torch
import hydra
import logging
from omegaconf import OmegaConf

# Import building function for model and dataset
from src.datasets.dataset_factory import instantiate_dataset
from src.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.base_tracker import BaseTracker
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.colors import COLORS

log = logging.getLogger(__name__)


def eval_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint):
    tracker.reset("val")
    loader = dataset.val_dataloader
    with Ctq(loader) as tq_val_loader:
        for data in tq_val_loader:
            with torch.no_grad():
                model.set_input(data, device)
                model.forward()

            tracker.track(model)
            tq_val_loader.set_postfix(**tracker.get_metrics(), color=COLORS.VAL_COLOR)

    tracker.print_summary()


def test_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint):

    loaders = dataset.test_dataloaders

    for loader in loaders:
        stage_name = loader.dataset.name
        tracker.reset(stage_name)
        with Ctq(loader) as tq_test_loader:
            for data in tq_test_loader:
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()

                tracker.track(model)
                tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

            tracker.print_summary()


def run(cfg, model, dataset: BaseDataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint):
    if dataset.has_val_loader:
        eval_epoch(model, dataset, device, tracker, checkpoint)

    test_epoch(model, dataset, device, tracker, checkpoint)


@hydra.main(config_path="conf/eval.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.checkpoint_dir, cfg.model_name, cfg.weight_name, strict=True)

    # Create model and datasets
    dataset = instantiate_dataset(checkpoint.data_config)
    model = checkpoint.create_model(dataset, weight_name=cfg.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    # Set dataloaders
    dataset.create_dataloaders(
        model, cfg.batch_size, cfg.shuffle, cfg.num_workers, cfg.precompute_multi_scale,
    )
    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    tracker: BaseTracker = dataset.get_tracker(model, dataset, False, False)

    # Run training / evaluation
    run(cfg, model, dataset, device, tracker, checkpoint)


if __name__ == "__main__":
    main()
