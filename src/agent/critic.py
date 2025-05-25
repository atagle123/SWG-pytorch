import os
import torch
import torch.nn.functional as F
import copy
import wandb
from tqdm import tqdm
from src.datasets.dataset import D4RL_Dataset
from src.utils.training import cycle
from src.utils.arrays import batch_to_device
from src.models.critic_models import Value_model, TwinQ
from src.utils.setup import DEVICE
from abc import ABC, abstractmethod


class Critic(ABC):
    def __init__(self, savepath):
        self.savepath = savepath

    @abstractmethod
    def value_update(self):
        pass

    @abstractmethod
    def q_update(self):
        pass

    def _log_info(self, info: dict[str, float], step: int, prefix: str, wandb_log: bool = False) -> None:
        info_str = " | ".join([f"{prefix}/{k}: {v:.4f}" for k, v in info.items()])
        print(f"{info_str} | (step {step})")

        if wandb_log:
            wandb.log({f"{prefix}/{k}": v for k, v in info.items()}, step=step)

    def train(self, n_steps, log_freq=10000, save_freq=100000, wandb_log=False):
        self.q_optimizer = torch.optim.Adam(
            self.q_model.parameters(), lr=self.train_params.q_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=self.train_params.q_lr
        )

        q_dataloader = cycle(
            torch.utils.data.DataLoader(
                self.q_dataset,
                batch_size=self.train_params.q_batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
        )

        for step in tqdm(range(1, n_steps + 1), smoothing=0.1):

            batch = next(q_dataloader)
            batch = batch_to_device(batch)

            v_loss, v = self.value_update(*batch)
            q_loss, qs = self.q_update(*batch)  # state, next_state, action, reward, terminals

            if step % log_freq == 0:
                q_mean = sum(q.mean() for q in qs) / len(qs)
                info = {
                        "v_loss": v_loss,
                        "v_mean": v.mean(),
                        "q_loss": q_loss,
                        "q_mean": q_mean}
                self._log_info(
                    info, step, prefix="train", wandb_log=wandb_log)

            if step % save_freq == 0 or step == 1:
                self.save(step=step)

    def update_q_target(self, tau):
        # Update the frozen target models
        for param, target_param in zip(
            self.q_model.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, step):
        """
        saves model
        """
        data = {
            "model": self.q_model.state_dict(),
            "target": self.q_target.state_dict(),
            "value": self.value_model.state_dict(),
        }

        savepath = os.path.join(self.critic_savepath, f"state_{step}.pt")
        torch.save(data, savepath)
        print(f"Saved model to {savepath}", flush=True)

    def load(self, step):
        """
        loads model from disk
        """
        print(f"loading step {step} critic model")
        loadpath = os.path.join(self.critic_savepath, f"state_{step}.pt")
        data = torch.load(loadpath)

        self.q_target.load_state_dict(data["target"])
        self.q_target.to(DEVICE)

        self.q_model.load_state_dict(data["model"])
        self.q_model.to(DEVICE)

        self.value_model.load_state_dict(data["value"])
        self.value_model.to(DEVICE)

        pass


class IQL_critic(Critic):
    def __init__(self, dataset_params, critic_params):

        self.critic_savepath = critic_params.critic_savepath

        self.q_dataset = D4RL_Dataset(**dataset_params)

        action_dim = self.q_dataset.action_dim
        state_dim = self.q_dataset.observation_dim

        self.q_model = TwinQ(
            action_dim=action_dim, state_dim=state_dim, **critic_params.q_model
        ).to(DEVICE)

        self.q_target = copy.deepcopy(self.q_model).requires_grad_(False).to(DEVICE)

        self.value_model = Value_model(
            state_dim=state_dim, **critic_params.value_model
        ).to(DEVICE)

        self.train_params = critic_params.training

        loss_map = {
            "expectile": self.expectile_loss,
            "quantile": self.quantile_loss,
            "linex": self.exponential_loss,
            "dice": self.dice_loss,
        }

        if self.train_params.objective in loss_map:
            self.iql_loss = loss_map[self.train_params.objective]
        else:
            print("Not supported objective")

    def value_update(self, states, next_states, actions, rewards, masks):
        with torch.no_grad():
            q_values = self.q_target(action=actions, state=states).detach()

        values = self.value_model(states)  # B,1

        loss = self.iql_loss(
            q_values, values, self.train_params.critic_hyperparam
        ).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=1.0)

        self.value_optimizer.step()

        return (loss, values)

    def q_update(self, states, next_states, actions, rewards, masks):
        """
        Calculate the Q-loss for a batch of states and actions

        Args:
            states (torch.tensor): B, State_Dim
            next_states (torch.tensor): B, State_Dim
            actions (torch.tensor): B, Action_Dim
            rewards (torch.tensor): B, 1
            terminals (torch.tensor): Boolean tensor B, 1

        Returns:
            loss: The computed Q-loss value for the given parameters.
        """

        with torch.no_grad():
            next_v = self.value_model(next_states)  # B,1
            targets = (
                rewards + masks * self.train_params.discount * next_v
            ).detach()  # B,1

        # Update Q function
        qs = self.q_model.both(action=actions, state=states)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)

        self.q_optimizer.step()

        # Update target
        self.update_q_target(tau=self.train_params.target_update)

        return (q_loss, qs)

    ### aux losses

    def expectile_loss(self, q_values, values, expectile=0.7):
        diff = q_values - values
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def quantile_loss(self, q_values, values, quantile=0.6):
        diff = q_values - values
        weight = torch.where(
            diff > 0,
            torch.tensor(quantile, device=DEVICE),
            torch.tensor(1 - quantile, device=DEVICE),
        )
        return weight * torch.abs(diff)

    def exp_w_clip(self, x, x0, mode="zero"):
        if mode == "zero":
            return torch.where(x < x0, torch.exp(x), torch.exp(x0))
        elif mode == "first":
            return torch.where(
                x < x0, torch.exp(x), torch.exp(x0) + torch.exp(x0) * (x - x0)
            )
        elif mode == "second":
            return torch.where(
                x < x0,
                torch.exp(x),
                torch.exp(x0)
                + torch.exp(x0) * (x - x0)
                + (torch.exp(x0) / 2) * ((x - x0) ** 2),
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def exponential_loss(
        self, q_values, values, beta, clip=torch.log(torch.tensor(6.0)), mode="zero"
    ):
        diff = q_values - values
        with torch.no_grad():  # Stop gradient for exp_diff
            exp_diff = self.exp_w_clip(diff * beta, clip, mode)
        return (exp_diff - 1) * diff

    def dice_loss(self, q_values, values, param=0.6):
        sp_term = (q_values - values) / param
        clipped_sp_term = torch.clamp(sp_term, max=1.0)

        residual_loss = torch.where(
            sp_term >= 0, sp_term**2 / 4 + sp_term, torch.exp(clipped_sp_term) - 1
        )
        value_loss = torch.mean(residual_loss + values / param)
        return value_loss
