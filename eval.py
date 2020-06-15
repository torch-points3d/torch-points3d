import hydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf/eval.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    print(cfg.pretty())

    trainer = Trainer(cfg)
    trainer.eval()


if __name__ == "__main__":
    main()
