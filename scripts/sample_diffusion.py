import os

from src.utils.setup import set_seed
from src.utils.visualization import visualize_data, visualize_chain
import numpy as np
from src.utils.setup import DEVICE
from src.utils.training import Trainer
from src.datasets import Toy_dataset_beta
from src.models.model import DiffusionMLP_beta
from src.models.diffusion import GaussianDiffusion_Beta
import hydra

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

dataset_name = "2spirals"

config_path = f"../logs/pretrained/toy/{dataset_name}/.hydra"

sample_params = {
    "guidance_scale": 1,
    "clip_denoised": True,
    "batch_size": 2000,
    "beta": 10,
    "return_chain": True,
}


@hydra.main(config_path=config_path, config_name="config", version_base="1.3")
def main(cfg):

    set_seed(cfg.seed)
    dataset = Toy_dataset_beta(**cfg.dataset)
    model = DiffusionMLP_beta(**cfg.model).to(DEVICE)
    diffusion = GaussianDiffusion_Beta(model=model, **cfg.diffusion).to(DEVICE)
    trainer = Trainer(
        diffusion_model=diffusion, dataset=dataset, **cfg.training
    )  # add log params

    trainer.load(step=200000)
    diffusion = trainer.ema_model

    diffusion.setup_sampling(**sample_params)

    diff_samples = diffusion()
    samples = diff_samples.sample.detach().cpu().numpy()
    chain = diff_samples.chains.detach().cpu().numpy()

    data = samples[:, :2]  # First two dimensions for data
    labels = samples[:, 2]  # Third dimension for labels
    labels = (np.log(labels) + sample_params["beta"]) / (
        sample_params["beta"] + 1e-6
    )  # unnormalize and get energy

    visualize_data(data=data, labels=labels)

    visualize_chain(chain=chain, beta=sample_params["beta"])


if __name__ == "__main__":
    main()
