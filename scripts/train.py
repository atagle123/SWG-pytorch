import os
import sys
import hydra
import logging
from src.utils.training import Trainer
from src.datasets import Toy_dataset_beta
from src.models.model import DiffusionMLP_beta
from src.models.diffusion import GaussianDiffusion_Beta
from src.utils.arrays import report_parameters
from src.utils.setup import set_seed
from src.utils.setup import DEVICE

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

config_path = f"../configs/toy"


@hydra.main(config_path=config_path, config_name="config", version_base="1.3")
def main(cfg):
    logger.info(f"Starting experiment with config: {cfg}")
    logger.info(f"Using DEVICE: {DEVICE}")

    set_seed(cfg.seed)

    os.makedirs(cfg.diffusion_results, exist_ok=True)

    dataset = Toy_dataset_beta(**cfg.dataset)
    model = DiffusionMLP_beta(**cfg.model).to(DEVICE)
    report_parameters(model)

    diffusion = GaussianDiffusion_Beta(model=model, **cfg.diffusion).to(DEVICE)
    trainer = Trainer(diffusion_model=diffusion, dataset=dataset, **cfg.training)

    trainer.train(n_train_steps=cfg.training.n_train_steps, **cfg.log_params)


if __name__ == "__main__":
    main()
