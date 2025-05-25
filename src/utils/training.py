import os
import copy
import torch
from tqdm import tqdm

from .arrays import batch_to_device


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        batch_size=1024,
        train_lr=2e-5,
        step_start_ema=10000,
        update_ema_every=10,
        results_dir="./results",
        **kwargs,
    ):

        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema

        self.dataset = dataset
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=train_lr, weight_decay=0.01
        )
        self.logdir = results_dir

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, step):
        if step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps, save_freq=20000, log_freq=10000):

        for step in tqdm(
            range(1, n_train_steps + 1), desc="Training"
        ):  # Wrap with tqdm

            batch = next(self.dataloader)
            batch = batch_to_device(batch)

            loss = self.model.loss(*batch)
            loss = loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if step % self.update_ema_every == 0:
                self.step_ema(step=step)

            if step % save_freq == 0:
                self.save(step)

            if step % log_freq == 0:
                print(f"{step}: {loss:8.4f}", flush=True)

    def save(self, step):
        """
        saves model and ema to disk;
        """
        data = {
            "step": step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f"state_{step}.pt")
        torch.save(data, savepath)

        print(f"[ utils/training ] Saved model to {savepath}", flush=True)

    def load(self, step):
        """
        loads model and ema from disk
        """
        if step == "latest":
            print(self.logdir)
            step = self.get_latest_step(self.logdir)

        loadpath = os.path.join(self.logdir, f"state_{step}.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def get_latest_step(self, loadpath):
        import glob

        states = glob.glob1(*loadpath, "state_*")
        latest_step = -1
        for state in states:
            step = int(state.replace("state_", "").replace(".pt", ""))
            latest_step = max(step, latest_step)
        return latest_step
