import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Sequence, Callable, Optional


# -----------------------------------------------------------------------------#
# ----------------------------- time embeddings -------------------------------#
# -----------------------------------------------------------------------------#


def divisible_by(numer, denom):
    return (numer % denom) == 0


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, learnable=True):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=learnable)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims: Sequence[int],
        activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
        activate_final: bool = False,
        use_layer_norm: bool = False,
        scale_final: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.scale_final = scale_final
        self.dropout_rate = dropout_rate

        # Layer normalization (if used)
        self.layer_norm = nn.LayerNorm(in_dim) if use_layer_norm else None

        # Create a list of layers
        layers = []

        for i, size in enumerate(self.hidden_dims):
            # Add linear layers
            layers.append(nn.Linear(in_dim, size))

            # Add activation function if not final layer or if activate_final is True
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                layers.append(self.activations)

            # Add dropout if specified
            if self.dropout_rate is not None and self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            in_dim = size  # Update the input dimension for the next layer

        # Combine the layers into a sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Forward pass through the layers
        return self.model(x)


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        act: Callable,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
    ):
        super(MLPResNetBlock, self).__init__()
        self.in_dim = in_dim
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        # Layers
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(in_dim) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        residual = x

        # Dropout (if specified)
        if self.dropout is not None:
            x = self.dropout(x) if training else x

        # Layer normalization (if specified)
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # MLP forward pass with activation
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        # Adjust residual if needed (for shape mismatch)
        if residual.shape != x.shape:
            residual = self.dense1(residual)  # Project residual to match shape

        # Return the residual connection
        return residual + x


class MLPResNet(nn.Module):
    """MLPResNet network."""

    def __init__(
        self,
        num_blocks: int,
        in_dim: int,
        out_dim: int,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Callable = F.relu,
    ):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        # Initial dense layer
        self.dense_input = nn.Linear(in_dim, hidden_dim)

        # MLP ResNet Blocks
        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    hidden_dim=hidden_dim * 4,
                    act=self.activations,
                    use_layer_norm=self.use_layer_norm,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output layer
        self.dense_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        # First dense layer
        x = self.dense_input(x)

        # Pass through MLP ResNet blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Apply activation and output layer
        x = self.activations(x)
        x = self.dense_output(x)

        return x


# -----------------------------------------------------------------------------#
# ---------------------------------- schedulers -------------------------------#
# -----------------------------------------------------------------------------#


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
    t = torch.arange(1, timesteps + 1, dtype=torch.float64)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas
