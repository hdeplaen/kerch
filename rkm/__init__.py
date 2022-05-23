import torch

__version__ = 0.1

import torch.cuda

PLOT_ENV = None

from . import kernel


@property
def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible on this machine.
    """
    return torch.cuda.is_available()
