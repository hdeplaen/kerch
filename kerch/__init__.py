__version__ = "v0.2"
__author__ = "HENRI DE PLAEN"
__credits__ = "KU Leuven"
__status__ = "alpha"
__date__ = "June 2022"

# GLOBAL MODULE-WIDE VARIABLES
_GLOBALS = {"PLOT_ENV": None,
            "LOG_LEVEL": 30,  # this corresponds to logging.WARNING
            }


# CHECK FUNCTIONALITIES
def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible on this machine.
    """
    import torch.cuda
    return torch.cuda.is_available()


# IMPORTS
from . import kernel    # ok (tested & documented)
from . import model     # alpha
from . import rkm       # beta
from . import dataset   # alpha
from . import plot      # alpha
from ._logger import set_log_level, get_log_level, _GLOBAL_LOGGER
from .utils import set_ftype, set_itype
