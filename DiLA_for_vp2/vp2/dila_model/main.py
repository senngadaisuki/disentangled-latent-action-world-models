import hydra
from omegaconf import DictConfig, OmegaConf
from train import Trainer

@hydra.main(config_path="configs", config_name="train_model", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Main training script for the hierarchical model.
    """
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    train()
