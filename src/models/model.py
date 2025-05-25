import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb


class DiffusionMLP(nn.Module):
    def __init__(self, data_dim=3, time_emb=4, hidden_dim=64, num_layers=3):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb),
            nn.Linear(time_emb, time_emb * 4),
            nn.Mish(),
            nn.Linear(time_emb * 4, time_emb),
        )

        # Create a series of fully connected layers
        layers = []
        layers.append(nn.Linear(data_dim + time_emb, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(
            nn.Linear(hidden_dim, data_dim)
        )  # Output layer to match input dimension

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x_combined = torch.cat((x, t), dim=-1)
        return self.model(x_combined)


class DiffusionMLP_beta(nn.Module):
    def __init__(self, data_dim=3, time_emb=32, hidden_dim=256, num_layers=5, **kwargs):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb),
            nn.Linear(time_emb, time_emb * 4),
            nn.Mish(),
            nn.Linear(time_emb * 4, time_emb),
        )

        self.beta_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb),
            nn.Linear(time_emb, time_emb * 4),
            nn.Mish(),
            nn.Linear(time_emb * 4, time_emb),
        )

        # Create a series of fully connected layers
        layers = []
        layers.append(nn.Linear(data_dim + time_emb * 2, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(
            nn.Linear(hidden_dim, data_dim)
        )  # Output layer to match input dimension

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, time, beta):

        t = self.time_mlp(time)
        b = self.beta_mlp(beta)
        t = torch.cat((t, b), dim=-1)
        x_combined = torch.cat((x, t), dim=-1)
        return self.model(x_combined)
