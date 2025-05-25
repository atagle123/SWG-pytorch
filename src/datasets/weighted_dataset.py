from .dataset import D4RL_Dataset
import torch
from collections import namedtuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from src.utils.setup import DEVICE

Weight_batch = namedtuple("Weight_batch", "actions_weight state")


ADV_MIN = -6.0
ADV_MAX = 4.0
EXP_MIN = 1e-3
EXP_MAX = 10.0
TEMP = 1


@dataclass
class Action_Weight_Dataset:
    actions_weight: np.ndarray
    observations: np.ndarray


class Weighted_Dataset(D4RL_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def build_weights(
        self,  # TODO add more weights...
        q_model,
        value_model,
        critic_hyperparam: float = 0.7,
        weights_function: str = "expectile",
        norm: bool = False,
    ):

        states = self.dataset.observations
        actions = self.dataset.actions

        states_batch = np.array_split(states, states.shape[0] // 256 + 1)
        actions_batch = np.array_split(actions, actions.shape[0] // 256 + 1)

        weights_list = []
        for states, actions in tqdm(zip(states_batch, actions_batch)):

            states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)

            qs = q_model(action=actions, state=states)  # (B, 1)
            vs = value_model(state=states)
            adv = qs - vs

            if weights_function == "expectile":
                weight = torch.where(
                    adv > 0, critic_hyperparam, 1 - critic_hyperparam
                )  # hyperparam of critic...

            elif weights_function == "quantile":  # TODO
                pass

            elif weights_function == "linex":
                weight = torch.abs(torch.exp(critic_hyperparam * adv) - 1) / torch.abs(
                    adv
                )
                weight = torch.clamp(weight, min=ADV_MIN, max=ADV_MAX)

            elif weights_function == "exponential":
                adv = torch.clamp(adv, min=ADV_MIN, max=ADV_MAX)
                weight = torch.exp(TEMP * adv)  # in this case e**bA
                weight = torch.clamp(weight, min=EXP_MIN, max=EXP_MAX)

            elif weights_function == "advantage":
                weight = adv

            elif weights_function == "dice":
                pi_residual = adv / critic_hyperparam

                if pi_residual.dim() == 1:
                    pi_residual = pi_residual.unsqueeze(1)

                weight = torch.where(
                    pi_residual >= 0, pi_residual / 2 + 1, torch.exp(pi_residual)
                )
                weight = torch.clamp(weight, min=1e-40, max=100)

            else:
                print("Not supported weights function")

            weights_list.append(weight)

        weights_tensor = torch.cat(weights_list, dim=0)
        weights_tensor = torch.nan_to_num(weights_tensor, nan=0.0)

        if norm:
            max = torch.max(weights_tensor)
            min = torch.min(weights_tensor)
            weights_tensor = (weights_tensor - min) / (max - min)

        min_weight = torch.min(weights_tensor)
        max_weight = torch.max(weights_tensor)
        mean_weight = torch.mean(weights_tensor)
        std_weights = torch.std(weights_tensor)

        print(f"min weight: {min_weight} | max_weight: {max_weight} | mean weight: {mean_weight} | std: {std_weights}")

        assert weights_tensor.shape == (len(self.dataset.next_observations), 1)

        self.weight_dataset = Action_Weight_Dataset(
            actions_weight=np.append(
                self.dataset.actions, weights_tensor.cpu().numpy(), axis=-1
            ),
            observations=self.dataset.observations,
        )
        del self.dataset

    def __len__(self):
        return len(self.weight_dataset.observations)

    def __getitem__(self, idx):

        return Weight_batch(
            actions_weight=self.weight_dataset.actions_weight[idx, :],
            state=self.weight_dataset.observations[idx, :],
        )
