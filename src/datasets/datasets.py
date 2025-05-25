import numpy as np
import torch
from collections import namedtuple
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


Batch = namedtuple("Batch", "sample")

Betas_batch = namedtuple("Betas_batch", "sample beta")


def create_swiss_roll(dataset_size):
    data = sklearn.datasets.make_swiss_roll(n_samples=dataset_size, noise=1.0)[0]
    data = data.astype("float64")[:, [0, 2]]
    data /= 5
    return data, np.sum(data**2, axis=-1, keepdims=True) / 9.0


def create_gaussians(dataset_size):
    scale = 4.0
    centers = [
        (0, 1),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1, 0),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (0, -1),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (1, 0),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    ]

    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    indexes = []
    for i in range(dataset_size):
        point = np.random.randn(2) * 0.5
        idx = np.random.randint(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        indexes.append(idx)
        dataset.append(point)
    dataset = np.array(dataset, dtype="float64")
    dataset /= 1.414
    return dataset, np.array(indexes, dtype="float64")[:, None] / 7.0


def create_2spirals(dataset_size):
    n = np.sqrt(np.random.rand(dataset_size // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(dataset_size // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(dataset_size // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x, np.clip((1 - np.concatenate([n, n]) / 10), 0, 1)


def create_moons(dataset_size):
    data, y = sklearn.datasets.make_moons(n_samples=dataset_size, noise=0.1)
    data = data.astype("float64")
    data = data * 2 + np.array([-1, -0.2])
    return data.astype(np.float64), (y > 0.5).astype(np.float64)[:, None]


def create_rings(dataset_size):
    n_samples4 = n_samples3 = n_samples2 = dataset_size // 4
    n_samples1 = dataset_size - n_samples4 - n_samples3 - n_samples2

    # so as not to have the first point = last point, we set endpoint=False
    linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
    linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
    linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
    linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

    circ4_x = np.cos(linspace4)
    circ4_y = np.sin(linspace4)
    circ3_x = np.cos(linspace4) * 0.75
    circ3_y = np.sin(linspace3) * 0.75
    circ2_x = np.cos(linspace2) * 0.5
    circ2_y = np.sin(linspace2) * 0.5
    circ1_x = np.cos(linspace1) * 0.25
    circ1_y = np.sin(linspace1) * 0.25

    X = (
        np.vstack(
            [
                np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
            ]
        ).T
        * 3.0
    )
    X = util_shuffle(X)

    center_dist = X[:, 0] ** 2 + X[:, 1] ** 2
    energy = np.zeros_like(center_dist)

    energy[(center_dist >= 8.5)] = 0.667
    energy[(center_dist >= 5.0) & (center_dist < 8.5)] = 0.333
    energy[(center_dist >= 2.0) & (center_dist < 5.0)] = 1.0
    energy[(center_dist < 2.0)] = 0.0

    # Add noise
    X = X + np.random.normal(scale=0.08, size=X.shape)

    return X.astype("float64"), energy[:, None]


def create_checkerboard(dataset_size):
    x1 = np.random.rand(dataset_size) * 4 - 2
    x2_ = np.random.rand(dataset_size) - np.random.randint(0, 2, dataset_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    points = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    points_x = points[:, 0]
    judger = ((points_x > 0) & (points_x <= 2)) | ((points_x <= -2))
    return points, judger.astype(np.float64)[:, None]


class Toy_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, beta=10, dataset_size=1000000):
        datasets_dict = {
            "8gaussians": create_gaussians,
            "2spirals": create_2spirals,
            "moons": create_moons,
            "swiss_roll": create_swiss_roll,
            "checkerboard": create_checkerboard,
            "rings": create_rings,
        }

        self.dataset_generator = datasets_dict[dataset_name]
        self.beta = beta
        self.dataset_size = dataset_size
        self.datas, self.energy = self.dataset_generator(dataset_size=dataset_size)
        self.datas = torch.tensor(self.datas, dtype=torch.float32)
        self.energy = torch.tensor(self.energy, dtype=torch.float32)
        self.energy = torch.exp(self.energy * beta)
        self.data_dim = 2
        self.get_params()
        self.normalize()

    def get_params(self):
        min = self.datas.min()
        max = self.datas.max()
        mean = self.datas.mean(axis=0)
        std = self.datas.std(axis=0)

        self.datas_params_dict = {"min": min, "max": max, "mean": mean, "std": std}

        min_energy = self.energy.min()
        max_energy = self.energy.max()
        mean_energy = self.energy.mean(axis=0)
        std_energy = self.energy.std(axis=0)

        self.energy_params_dict = {
            "min": min_energy,
            "max": max_energy,
            "mean": mean_energy,
            "std": std_energy,
        }

    def normalize(self):
        min = self.datas_params_dict["min"]
        max = self.datas_params_dict["max"]
        mean = self.datas_params_dict["mean"]
        std = self.datas_params_dict["std"]
        # Z-score normalization
        # self.datas = (self.datas - mean) / std
        self.datas = (self.datas - min) / (max - min)

        # Z-score normalization
        # self.datas = (self.datas - mean) / std
        self.datas = self.datas * 2 - 1

        min_energy = self.energy_params_dict["min"]
        max_energy = self.energy_params_dict["max"]
        mean_energy = self.energy_params_dict["mean"]
        std_energy = self.energy_params_dict["std"]
        # self.energy=(self.energy - min_energy) / (max_energy - min_energy)
        self.energy = self.energy / (max_energy - min_energy)
        # self.energy=self.energy*2-1

    def __getitem__(self, index):
        sample = torch.cat((self.datas[index], self.energy[index]), dim=0)
        return Batch(sample)

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_size


class Toy_dataset_beta(torch.utils.data.Dataset):
    def __init__(self, dataset_name, betas_range=[0, 25], dataset_size=1000000):
        datasets_dict = {
            "8gaussians": create_gaussians,
            "2spirals": create_2spirals,
            "moons": create_moons,
            "swiss_roll": create_swiss_roll,
            "checkerboard": create_checkerboard,
            "rings": create_rings,
        }

        self.dataset_generator = datasets_dict[dataset_name]
        self.betas_range = betas_range
        self.dataset_size = dataset_size
        self.datas, self.energy = self.dataset_generator(dataset_size=dataset_size)
        self.datas = torch.tensor(self.datas, dtype=torch.float32)
        self.energy = torch.tensor(self.energy, dtype=torch.float32)
        self.energy = torch.exp(self.energy)
        self.data_dim = 2
        self.get_params()
        self.normalize()

    def get_params(self):
        min = self.datas.min()
        max = self.datas.max()
        mean = self.datas.mean(axis=0)
        std = self.datas.std(axis=0)

        self.datas_params_dict = {"min": min, "max": max, "mean": mean, "std": std}

        min_energy = self.energy.min()
        max_energy = self.energy.max()
        mean_energy = self.energy.mean(axis=0)
        std_energy = self.energy.std(axis=0)

        self.energy_params_dict = {
            "min": min_energy,
            "max": max_energy,
            "mean": mean_energy,
            "std": std_energy,
        }

    def unnormalize(self, data, labels, beta):
        min = self.datas_params_dict["min"]
        max = self.datas_params_dict["max"]

        data = (data + 1) / 2
        data = data * (max - min) + min
        e = torch.tensor(torch.e)
        labels = torch.log(labels) / (beta + 1e-6) + 1
        return (data, labels)

    def normalize(self):
        min = self.datas_params_dict["min"]
        max = self.datas_params_dict["max"]
        mean = self.datas_params_dict["mean"]
        std = self.datas_params_dict["std"]
        # Z-score normalization
        # self.datas = (self.datas - mean) / std

        self.datas = (self.datas - min) / (max - min)

        self.datas = self.datas * 2 - 1

        min_energy = self.energy_params_dict["min"]
        max_energy = self.energy_params_dict["max"]
        mean_energy = self.energy_params_dict["mean"]
        std_energy = self.energy_params_dict["std"]

        e = torch.tensor(torch.e)

        # Powers from beta_min to beta_max
        powers = torch.arange(self.betas_range[0], self.betas_range[1])

        # Compute e raised to the powers
        result = e**powers  # e**b
        self.energy_max = (
            1 / result
        )  # list with the maximum exp energy possible or each beta, considering E is between [0,1]

    def __getitem__(self, index):

        sampled_beta = torch.randint(
            low=self.betas_range[0], high=self.betas_range[1], size=(1,)
        )
        energy_beta = torch.pow(self.energy[index], sampled_beta)

        energy_beta = energy_beta * self.energy_max[sampled_beta]
        sample = torch.cat((self.datas[index], energy_beta), dim=0)
        return Betas_batch(sample, sampled_beta.squeeze())

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_size
