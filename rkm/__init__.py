__version__ = "0.1.2"
__author__ = "HENRI DE PLAEN"
__credits__ = "KU Leuven"

_PLOT_ENV = None


def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible on this machine.
    """
    import torch.cuda
    return torch.cuda.is_available()


# LOGGING
import logging

_LOG_LEVEL = logging.DEBUG

# IMPORTS
from . import kernel
from . import model
from .utils import _logger, set_log_level, get_log_level, set_ftype, set_itype
