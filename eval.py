import hydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    trainer.eval()


if __name__ == "__main__":
    main()
