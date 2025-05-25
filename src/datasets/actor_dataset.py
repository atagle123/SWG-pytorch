from .dataset import D4RL_Dataset
from collections import namedtuple

Actions_batch = namedtuple("Actions_batch", "action state")


class Actor_Dataset(D4RL_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.dataset.observations)

    def __getitem__(self, idx):

        return Actions_batch(
            action=self.dataset.actions[idx, :], state=self.dataset.observations[idx, :]
        )
