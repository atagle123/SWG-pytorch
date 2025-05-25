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
import scipy
import os
import sys
import scipy
import numpy as np
from src.utils.setup import set_seed
from src.datasets.datasets import (
    create_2spirals,
    create_checkerboard,
    create_gaussians,
    create_moons,
    create_rings,
    create_swiss_roll,
)
import matplotlib.pyplot as plt


# Sample function for dataset and diffusion plotting
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
        labels = torch.zeros_like(labels)

    return data.cpu().numpy(), labels.cpu().numpy()


def load_diffusion(dataset):
    config_path = f"../logs/pretrained/toy/{dataset}/.hydra"
    hydra.initialize(config_path=config_path, version_base="1.3")
    cfg = hydra.compose(config_name="config")

    dataset = Toy_dataset_beta(**cfg.dataset)
    model = DiffusionMLP_beta(**cfg.model).to(DEVICE)
    diffusion = GaussianDiffusion_Beta(model=model, **cfg.diffusion).to(DEVICE)
    trainer = Trainer(diffusion_model=diffusion, dataset=dataset, **cfg.training)

    trainer.load(step=200000)
    diffusion = trainer.ema_model
    return diffusion, dataset


def energy_sample(dataset_name, beta, sample_per_state=2000):
    datasets_dict = {
        "8gaussians": create_gaussians,
        "2spirals": create_2spirals,
        "moons": create_moons,
        "swiss_roll": create_swiss_roll,
        "checkerboard": create_checkerboard,
        "rings": create_rings,
    }

    dataset_generator = datasets_dict[dataset_name]
    data, energy = dataset_generator(dataset_size=1000 * sample_per_state)
    beta_energy = energy * beta
    index = np.random.choice(
        1000 * sample_per_state,
        p=scipy.special.softmax(beta_energy).squeeze(),
        size=sample_per_state,
        replace=False,
    )
    data = data[index]
    energy = energy[index]
    return data, energy


def sample_all():
    for dataset_name in [
        "2spirals"
    ]:  # ["rings", "8gaussians", "swiss_roll","rings","checkerboard","moons"]:
        fig, axes = plt.subplots(2, 5, figsize=(16, 6))
        betas = [0, 1, 5, 10, 15]
        GlobalHydra.instance().clear()
        diffusion, dataset = load_diffusion(dataset_name)

        for i, beta in enumerate(betas):
            ax1 = axes[0, i]
            data, e = sample_for_dataset(diffusion, beta, dataset)
            ax1.set_aspect("equal", adjustable="box")
            ax1.set_xlim(-4.5, 4.5)
            ax1.set_ylim(-4.5, 4.5)
            mappable = ax1.scatter(
                data[:, 0],
                data[:, 1],
                s=1,
                c=e,
                cmap="winter",
                vmin=0,
                vmax=1,
                rasterized=True,
            )
            ax1.set_title(r"$\beta={}$".format(beta), fontsize=24)

            ax1.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            ax1.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )

            ax2 = axes[1, i]
            data, e = energy_sample(dataset_name, beta, sample_per_state=2000)
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_xlim(-4.5, 4.5)
            ax2.set_ylim(-4.5, 4.5)
            ax2.scatter(
                data[:, 0],
                data[:, 1],
                s=1,
                c=e,
                cmap="winter",
                vmin=0,
                vmax=1,
                rasterized=True,
            )

            ax2.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            ax2.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )

        axes[0, 0].text(
            -6.2,
            -3,
            "SWG (ours)",
            fontsize=24,
            ha="center",
            va="bottom",
            fontname="serif",
            rotation=90,
        )
        axes[1, 0].text(
            -6.2,
            4,
            "Ground truth",
            fontsize=24,
            ha="center",
            va="top",
            fontname="serif",
            rotation=90,
        )

        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"$-\xi(a)$", fontsize=24, rotation=0, labelpad=20, ha="center")

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(f"{dataset_name}_combined_plot.pdf", dpi=300)


def main():
    set_seed(42)
    sample_all()
    plt.show()


if __name__ == "__main__":
    main()
