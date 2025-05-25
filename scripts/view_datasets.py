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
    # scatter ground truth
    for dataset in [
        "2spirals"
    ]:  # , "8gaussians", "swiss_roll","rings","checkerboard","moons"]:
        plt.figure(figsize=(12, 3.0))
        betas = [0, 1, 5, 15]
        axes = []
        for i, beta in enumerate(betas):
            plt.subplot(1, len(betas), i + 1)
            data, e = energy_sample(dataset, beta, sample_per_state=2000)
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
                    vmax=1,
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
        colorbar = plt.gcf().colorbar(
            mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12
        )
        colorbar.set_label(
            r"$\frac{\log(w)}{\beta}$",
            fontsize=14,
            rotation=0,
            labelpad=10,
            ha="center",
        )  # Fixed label
        plt.savefig(f"{dataset}_gt.pdf", dpi=300)  # Saves as a PDF


def main():
    set_seed(42)
    sample_all()
    plt.show()


if __name__ == "__main__":
    main()
