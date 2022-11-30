__version__ = "0.2.2"
__author__ = "HENRI DE PLAEN"
__credits__ = "KU Leuven"
__status__ = "alpha"
__date__ = "November 2022"

# GLOBAL MODULE-WIDE VARIABLES
_GLOBALS = {"PLOT_ENV": None,
            "LOG_LEVEL": 30,  # this corresponds to logging.WARNING
            }

# IMPORTS
from . import kernel    # ok (tested & documented)
from . import rkm  # beta
from . import model  # alpha
# from . import dataset  # alpha
from . import plot  # alpha
from . import opt  # alpha
from ._logger import set_log_level, get_log_level, _GLOBAL_LOGGER
from .utils import set_ftype, set_itype, FTYPE, ITYPE

# CHECK FUNCTIONALITIES
def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible on this machine.
    """
    import torch.cuda
    if torch.cuda.is_available():
        _GLOBAL_LOGGER.info("Using CUDA version " + torch.version.cuda)
        return True
    return False