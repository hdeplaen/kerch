"""
This abstract class has only one purpose, add a self._log property meant to log various information across the toolbox.
Doing it this way allows to get the name of the class instantiated and print more precise messages.

Author: HENRI DE PLAEN
Date: June 2022
"""

import logging
from abc import ABCMeta, abstractmethod

from . import _GLOBALS

_LOGGING_FORMAT = "KerPy %(levelname)s [%(name)s]: %(message)s"

_kerpy_format = logging.Formatter(_LOGGING_FORMAT)
_kerpy_handler = logging.StreamHandler()
_kerpy_handler.setFormatter(_kerpy_format)

class _logger(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self._log = logging.getLogger(name=self.__class__.__name__)
        self._log.addHandler(_kerpy_handler)
        self._log.setLevel(_GLOBALS["LOG_LEVEL"])

def set_log_level(level: int):
    r"""
    Sets the logging level of the toolbox. The best is to use a value of the logging module.
    For example

    .. code-block::
        import kerpy
        import logging

        kerpy.set_log_level(logging.DEBUG)


    .. note::
        Changing the log value does not affect the already instantiated objects.


    """
    _GLOBALS["LOG_LEVEL"] = level


def get_log_level() -> str:
    r"""
    Returns the logging level of the toolbox.
    """
    return logging.getLevelName(_GLOBALS["LOG_LEVEL"])
