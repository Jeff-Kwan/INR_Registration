"""Device eligibility for optional native CUDA/Triton inference and scoring."""
from functools import lru_cache
from importlib.util import find_spec

import torch


@lru_cache(maxsize=None)
def supported(device):
    return (device.type == 'cuda' and torch.version.cuda is not None
            and torch.cuda.get_device_capability(device)[0] >= 8
            and find_spec('triton') is not None)
