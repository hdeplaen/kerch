# coding=utf-8
__version__ = "0.3"
__author__ = "HENRI DE PLAEN"
__credits__ = "KU Leuven"
__status__ = "beta"
__date__ = "January 2024"

# GLOBAL MODULE-WIDE VARIABLES
_GLOBALS = {"PLOT_ENV": None,
            "LOG_LEVEL": 30,  # this corresponds to logging.WARNING
            }

__all__ = ['__version__', '__author__', '__credits__', '__status__', '__date__', 'kernel', 'level', 'model', 'data',
           'train', 'opt', 'set_log_level', 'get_log_level', 'gpu_available', 'set_ftype', 'set_itype',
           'DEFAULT_KERNEL_TYPE', 'DEFAULT_CACHE_LEVEL']


# IMPORTS
from . import kernel as kernel  # ok (tested & documented)
from . import level as level  # beta
from . import model as model  # beta
from . import data as data  # beta
from . import train as train  # alpha
from . import plot as plot             # alpha
from . import opt as opt  # beta
from .module._Logger import (set_log_level as set_log_level,
                             get_log_level as get_log_level)
from .utils import (gpu_available as gpu_available,
                    set_ftype as set_ftype,
                    set_itype as set_itype,
                    set_eps as set_eps,
                    DEFAULT_KERNEL_TYPE as DEFAULT_KERNEL_TYPE,
                    DEFAULT_CACHE_LEVEL as DEFAULT_CACHE_LEVEL)
