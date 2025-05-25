import torch
import torch.nn as nn
from .helpers import (
    RandomOrLearnedSinusoidalPosEmb,
    MLP,
    MLPResNet)

class ScoreModel(nn.Module):
    def __init__(
        self,
        data_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        time_emb: int = 128,
        num_blocks: int = 3,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.time_encoder = RandomOrLearnedSinusoidalPosEmb(
            dim=time_emb, learnable=True
        )
        self.cond_encoder = MLP(
            in_dim=time_emb + 1,
            hidden_dims=(time_emb * 2, time_emb * 2),
            activations=nn.SiLU(),
            activate_final=False,
        )

        self.base_model = MLPResNet(
            in_dim=time_emb * 2 + state_dim + data_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            out_dim=data_dim,
            activations=nn.SiLU(),
        )

    def forward(self, x, state, time, training=True):
        t = self.time_encoder(time)
        cond_emb = self.cond_encoder(t)
        reverse_input = torch.cat([x, state, cond_emb], dim=-1)
        out = self.base_model(reverse_input, training)

        return out