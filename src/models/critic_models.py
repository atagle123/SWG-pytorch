import torch
import torch.nn as nn


def mlp(
    dims: list,
    activation=nn.ReLU,
    activate_final: bool = False,
    use_layer_norm: bool = False,
):
    n_dims = len(dims)
    assert n_dims >= 2, "MLP requires at least two dims (input and output)"

    if use_layer_norm:
        layers = [nn.LayerNorm(dims[0])]
    else:
        layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())

    layers.append(nn.Linear(dims[-2], dims[-1]))
    if activate_final:
        layers.append(activation())

    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class Value_model(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, use_layer_norm=use_layer_norm)

    def forward(self, state):
        return self.v(state)
