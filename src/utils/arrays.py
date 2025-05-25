import numpy as np
import torch
from .setup import DEVICE

DTYPE = torch.float


def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


# -----------------------------------------------------------------------------#
# ------------------------------ numpy <--> torch -----------------------------#
# -----------------------------------------------------------------------------#


def to_device(x):
    if torch.is_tensor(x):
        x = x.float()
        return x.to(DEVICE, non_blocking=True)
    else:
        raise RuntimeError(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch):
    vals = [to_device(getattr(batch, field)) for field in batch._fields]
    return type(batch)(*vals)


def _to_str(num):
    if num >= 1e6:
        return f"{(num/1e6):.2f} M"
    else:
        return f"{(num/1e3):.2f} k"


# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters
