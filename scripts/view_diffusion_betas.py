from src.utils.setup import set_seed
import matplotlib.pyplot as plt

import numpy as np
from src.utils.setup import DEVICE
from src.utils.training import Trainer
from src.datasets import Toy_dataset_beta
from src.models.model import DiffusionMLP_beta
from src.models.diffusion import GaussianDiffusion_Beta
import hydra
from hydra.core.global_hydra import GlobalHydra
import torch


def sample_for_dataset(diffusion, beta, dataset):
    sample_params = {"guidance_scale": 1, "clip_denoised": True, "batch_size": 2000}

    sample_params["beta"] = beta
    if beta == 0:
        sample_params["guidance_scale"] = 0
        sample_params["beta"] = 3

    diffusion.setup_sampling(**sample_params)

    samples = diffusion().sample.detach()

    data = samples[:, :2]  # First two dimensions for data
    labels = samples[:, 2]  # Third dimension for labels
    data, labels = dataset.unnormalize(data, labels, beta)
    if beta == 0:
        labels = tensor = torch.zeros_like(labels)

    return (data.cpu().numpy(), labels.cpu().numpy())


def load_diffusion(dataset):
    config_path = f"../logs/pretrained/toy/{dataset}/.hydra"

    hydra.initialize(config_path=config_path, version_base="1.3")

    cfg = hydra.compose(config_name="config")

    dataset = Toy_dataset_beta(**cfg.dataset)
    model = DiffusionMLP_beta(**cfg.model).to(DEVICE)
    diffusion = GaussianDiffusion_Beta(model=model, **cfg.diffusion).to(DEVICE)
    trainer = Trainer(diffusion_model=diffusion, dataset=dataset, **cfg.training)

    trainer.load(step=120000)
    diffusion = trainer.ema_model
    return (diffusion, dataset)


def main():
    set_seed(123)
    for dataset_name in [
        "2spirals"
    ]:  # ["moons", "8gaussians", "swiss_roll", "rings", "checkerboard", "2spirals"]:
        plt.figure(figsize=(12, 3.0))
        betas = [0, 1, 5, 15]
        axes = []
        GlobalHydra.instance().clear()
        diffusion, dataset = load_diffusion(dataset_name)

        for i, beta in enumerate(betas):
            plt.subplot(1, len(betas), i + 1)
            data, e = sample_for_dataset(diffusion, beta, dataset)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(-4.5, 4.5)
            plt.ylim(-4.5, 4.5)
            if i == 0:
                mappable = plt.scatter(
                    data[:, 0],
                    data[:, 1],
                    s=1,
                    c=e,
                    cmap="winter",
                    vmin=0,
                    vmax=100000,
                    rasterized=True,
                )
                plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            else:
                plt.scatter(
                    data[:, 0],
                    data[:, 1],
                    s=1,
                    c=e,
                    cmap="winter",
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
                plt.yticks(
                    ticks=[-4, -2, 0, 2, 4], labels=[None, None, None, None, None]
                )
            axes.append(plt.gca())
            plt.xticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            plt.title(r"$\beta={}$".format(beta))
        plt.tight_layout()
        plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.savefig(f"{dataset_name}.pdf", dpi=300)  # Saves as a PDF


if __name__ == "__main__":
    main()
    plt.show()
