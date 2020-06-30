import shutil
import os
import subprocess


class Wandb:
    IS_ACTIVE = False

    @staticmethod
    def _set_to_wandb_args(wandb_args, cfg, name):
        var = getattr(cfg.wandb, name, None)
        if var:
            wandb_args[name] = var

    @staticmethod
    def launch(cfg, launch: bool):
        if launch:
            import wandb

            Wandb.IS_ACTIVE = True

            model_config = getattr(cfg.models, cfg.model_name, None)
            model_class = getattr(model_config, "class")
            tested_dataset_class = getattr(cfg.data, "class")
            otimizer_class = getattr(cfg.training.optim.optimizer, "class")
            scheduler_class = getattr(cfg.lr_scheduler, "class")
            tags = [
                cfg.model_name,
                model_class.split(".")[0],
                tested_dataset_class,
                otimizer_class,
                scheduler_class,
            ]

            wandb_args = {}
            wandb_args["project"] = cfg.wandb.project
            wandb_args["tags"] = tags
            try:
                wandb_args["config"] = {
                    "run_path": os.getcwd(),
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip(),
                }
            except:
                wandb_args["config"] = {
                    "run_path": os.getcwd(),
                    "commit": "n/a",
                }
            Wandb._set_to_wandb_args(wandb_args, cfg, "name")
            Wandb._set_to_wandb_args(wandb_args, cfg, "entity")
            Wandb._set_to_wandb_args(wandb_args, cfg, "notes")

            wandb.init(**wandb_args)
            shutil.copyfile(
                os.path.join(os.getcwd(), ".hydra/config.yaml"), os.path.join(os.getcwd(), ".hydra/hydra-config.yaml")
            )
            wandb.save(os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"))
            wandb.save(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))

    @staticmethod
    def add_file(file_path: str):
        if not Wandb.IS_ACTIVE:
            raise RuntimeError("wandb is inactive, please launch first.")
        import wandb

        filename = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(wandb.run.dir, filename))
