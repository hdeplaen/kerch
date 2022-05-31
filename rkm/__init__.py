__version__ = "0.1.2"
__author__ = "HENRI DE PLAEN"
__credits__ = "KU Leuven"

PLOT_ENV = None

def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible on this machine.
    """
    import torch.cuda
    return torch.cuda.is_available()

# IMPORTS
from . import kernel
