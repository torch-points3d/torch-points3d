import torch
import hydra
import logging
from omegaconf import OmegaConf

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.utils.colors import COLORS

log = logging.getLogger(__name__)


def eval_epoch(
    model: BaseModel,
    dataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    voting_runs=1,
    tracker_options={},
):
    tracker.reset("val")
    loader = dataset.val_dataloader
    for i in range(voting_runs):
        with Ctq(loader) as tq_val_loader:
            for data in tq_val_loader:
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()

                tracker.track(model, data=data, **tracker_options)
                tq_val_loader.set_postfix(**tracker.get_metrics(), color=COLORS.VAL_COLOR)

    tracker.finalise(**tracker_options)
    tracker.print_summary()


def test_epoch(
    model: BaseModel,
    dataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    voting_runs=1,
    tracker_options={},
):

    loaders = dataset.test_dataloaders

    for loader in loaders:
        stage_name = loader.dataset.name
        if not loader.has_labels and not tracker_options["make_submission"]:  # No label, no submission -> do nothing
            log.warning("No forward will be run on dataset %s." % stage_name)
            continue

        tracker.reset(stage_name)
        for i in range(voting_runs):
            with Ctq(loader) as tq_test_loader:
                for data in tq_test_loader:
                    with torch.no_grad():
                        model.set_input(data, device)
                        model.forward()

                    tracker.track(model, data=data, **tracker_options)
                    tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

        tracker.finalise(**tracker_options)
        tracker.print_summary()


def run(
    cfg,
    model,
    dataset: BaseDataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    voting_runs=1,
    tracker_options={},
):
    if dataset.has_val_loader:
        eval_epoch(
            model, dataset, device, tracker, checkpoint, voting_runs=voting_runs, tracker_options=tracker_options
        )

    if dataset.has_test_loaders:
        test_epoch(
            model, dataset, device, tracker, checkpoint, voting_runs=voting_runs, tracker_options=tracker_options,
        )


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

    tracker: BaseTracker = dataset.get_tracker(False, False)

    # Run training / evaluation
    run(
        cfg,
        model,
        dataset,
        device,
        tracker,
        checkpoint,
        voting_runs=cfg.voting_runs,
        tracker_options=cfg.tracker_options,
    )


if __name__ == "__main__":
    main()
