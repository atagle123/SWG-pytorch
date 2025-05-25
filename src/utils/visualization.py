import matplotlib.pyplot as plt
import numpy as np


def visualize_data(data, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=1,
        c=labels,
        cmap="winter",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    plt.colorbar(label="Energy")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Visualization of Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.tight_layout()
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def visualize_data_simple(data, labels, ax=None):
    # If no axis object is provided, create a new figure
    if ax is None:
        ax = plt.gca()

    sc = ax.scatter(
        data[:, 0],
        data[:, 1],
        s=1,
        c=labels,
        cmap="winter",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Visualization of Points")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    return sc


def visualize_chain(chain, beta):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    data = chain[:, 0, :2]
    labels = chain[:, 0, 2]
    labels = (np.log(labels) + beta) / (beta + 1e-6)  # unnormalize and get energy

    sc1 = visualize_data_simple(data, labels, axes[0, 0])
    data = chain[:, 35, :2]
    labels = chain[:, 35, 2]
    labels = (np.log(labels) + beta) / (beta + 1e-6)  # unnormalize and get energy

    sc2 = visualize_data_simple(data, labels, axes[0, 1])
    data = chain[:, 70, :2]
    labels = chain[:, 70, 2]
    labels = (np.log(labels) + beta) / (beta + 1e-6)  # unnormalize and get energy

    sc3 = visualize_data_simple(data, labels, axes[1, 0])
    data = chain[:, -1, :2]
    labels = chain[:, -1, 2]
    labels = (np.log(labels) + beta) / (beta + 1e-6)  # unnormalize and get energy

    sc4 = visualize_data_simple(data, labels, axes[1, 1])

    # Add colorbars to each subplot
    fig.colorbar(sc1, ax=axes[0, 0], label="Energy")
    fig.colorbar(sc2, ax=axes[0, 1], label="Energy")
    fig.colorbar(sc3, ax=axes[1, 0], label="Energy")
    fig.colorbar(sc4, ax=axes[1, 1], label="Energy")
    plt.show()
