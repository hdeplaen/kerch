import torch

__version__ = 0.1
__author__ = 'HENRI DE PLAEN'
__credits__ = 'KU Leuven'

import torch.cuda

PLOT_ENV = None

from . import kernel


@property
def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible on this machine.
    """
    return torch.cuda.is_available()
