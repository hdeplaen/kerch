# coding=utf-8
__version__ = "0.3"
__author__ = "HENRI DE PLAEN"
__credits__ = "KU Leuven"
__status__ = "beta"
__date__ = "January 2024"
__license__ = "LGPL-3.0"

# GLOBAL MODULE-WIDE VARIABLES
_GLOBALS = {"PLOT_ENV": None,
            "LOG_LEVEL": 30,  # this corresponds to logging.WARNING
            }

__all__ = ['__version__', '__author__', '__credits__', '__status__', '__date__', '__license__',
           'kernel', 'level', 'model', 'data', 'train', 'opt', 'set_logging_level', 'get_logging_level', 'gpu_available',
           'set_ftype', 'set_itype', 'DEFAULT_KERNEL_TYPE', 'DEFAULT_CACHE_LEVEL', 'FTYPE', 'ITYPE']


# IMPORTS
from . import kernel as kernel  # ok (tested & documented)
from . import level as level  # beta
from . import model as model  # beta
from . import data as data  # beta
from . import train as train  # alpha
from . import monitor as monitor             # alpha
from . import plot as plot
from . import opt as opt  # beta
from . import script as script # alpha
from .feature.logger import (set_logging_level as set_logging_level,
                            get_logging_level as get_logging_level)
from .utils import (gpu_available as gpu_available,
                    FTYPE as FTYPE,
                    ITYPE as ITYPE,
                    set_ftype as set_ftype,
                    set_itype as set_itype,
                    set_eps as set_eps,
                    DEFAULT_KERNEL_TYPE as DEFAULT_KERNEL_TYPE,
                    DEFAULT_CACHE_LEVEL as DEFAULT_CACHE_LEVEL)
