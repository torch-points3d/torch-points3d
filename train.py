import hydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    
    trainer = Trainer(cfg)
    trainer.train()
    return 0


if __name__ == "__main__":
    main()
