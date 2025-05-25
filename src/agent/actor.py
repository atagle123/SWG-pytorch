import os
import wandb
import copy
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets import Weighted_Dataset
from src.models.diffusion import SelfWeighted_GaussianDiffusion

from src.models.diffusion_net import ScoreModel
from src.utils.training import cycle, EMA
from src.utils.arrays import batch_to_device, report_parameters
from src.utils.setup import DEVICE


class Actor:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.diffusion_model.state_dict())

    def step_ema(self, step):
        if step < self.train_params.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_model)

    ### saving and loading
    def save(self, step):
        """
        saves model and ema to disk;
        """
        data = {
            "model": self.diffusion_model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.actor_savepath, f"state_{step}.pt")
        torch.save(data, savepath)

        print(f"Saved model to {savepath}", flush=True)

    def load(self, step):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.actor_savepath, f"state_{step}.pt")
        data = torch.load(loadpath)

        self.diffusion_model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def load_latest_step(self, step):
        if step == "latest":
            step = self._get_latest_step(self.actor_savepath)
        self.load(step)

    def _get_latest_step(self, loadpath):
        import glob

        states = glob.glob1(*loadpath, "state_*")
        latest_step = -1
        for state in states:
            step = int(state.replace("state_", "").replace(".pt", ""))
            latest_step = max(step, latest_step)
        return latest_step
    
    def _log_info(self, info: dict[str, float], step: int, prefix: str, wandb_log: bool = False) -> None:
        info_str = " | ".join([f"{prefix}/{k}: {v:.4f}" for k, v in info.items()])
        print(f"{info_str} | (step {step})")

        if wandb_log:
            wandb.log({f"{prefix}/{k}": v for k, v in info.items()}, step=step)


class Weighted_actor(Actor):
    def __init__(self, dataset_params, actor_params):

        self.actor_savepath = actor_params.actor_savepath
        self.dataset = Weighted_Dataset(**dataset_params)

        action_dim = self.dataset.action_dim
        state_dim = self.dataset.observation_dim

        model = ScoreModel(
            data_dim=action_dim + 1, state_dim=state_dim, **actor_params.diffusion_model
        ).to(DEVICE)

        report_parameters(model)
        self.diffusion_model = SelfWeighted_GaussianDiffusion(
            model=model, data_dim=action_dim + 1, **actor_params.diffusion
        ).to(DEVICE)

        self.train_params = actor_params.training

        self.actor_params = actor_params
        self.ema = EMA(self.train_params.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)
        self.reset_parameters()

    def train(self, n_steps, save_freq=100000, log_freq=10000, wandb_log=False):

        dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.train_params.train_batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
            )
        )

        optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=self.train_params.train_lr,
            weight_decay=self.train_params.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=self.train_params.eta_min
        )

        for step in tqdm(range(1, n_steps + 1), smoothing=0.1):

            batch = next(dataloader)
            batch = batch_to_device(batch)

            loss, info = self.diffusion_model.loss(*batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            if step % self.train_params.update_ema_every == 0:
                with torch.no_grad():
                    self.step_ema(step)

            if step % save_freq == 0 or step == 1:
                self.save(step=step)

            if step % log_freq == 0:
            
                self._log_info(
                    info,
                    step=step,
                    prefix="train",
                    wandb_log=wandb_log,
                )
