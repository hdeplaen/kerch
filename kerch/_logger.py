"""
This abstract class has only one purpose, add a self._log property meant to log various information across the toolbox.
Doing it this way allows to get the name of the class instantiated and print more precise messages.

Author: HENRI DE PLAEN
Date: June 2022
"""

import logging
from abc import ABCMeta, abstractmethod

from . import _GLOBALS, utils

_LOGGING_FORMAT = "Kerch %(levelname)s [%(name)s]: %(message)s [%(pathname)s:%(lineno)d]"

_kerch_format = logging.Formatter(_LOGGING_FORMAT)
_kerch_handler = logging.StreamHandler()
_kerch_handler.setFormatter(_kerch_format)


class _Logger(metaclass=ABCMeta):
    @utils.kwargs_decorator({
        "log_level": None,
        "name": None
    })
    def __init__(self, *args, **kwargs):
        self._name = kwargs["name"]
        class_name = self.__class__.__name__
        if self._name is not None and class_name != "_Logger":
            log_name = self._name + ' ' + class_name
        elif self._name is not None:
            log_name = self._name
        else:
            log_name = class_name
        self._log = logging.getLogger(name=log_name)
        self._log.addHandler(_kerch_handler)
        self.set_log_level(kwargs["log_level"])

    def set_log_level(self, level: int = None) -> int:
        r"""
        Sets a specific log Level to this object. It serves as a way to use specific log Level for a specific class,
        different than the current general KerPy log Level.

        :param level: If the value is ``None``, the current general KerPy log Level will be used (WARNING if not
            specified otherwise)., defaults to ``None``.
        :param type: int, optional
        """
        if level is None:
            level = _GLOBALS["LOG_LEVEL"]
        self._log.setLevel(level)
        return level

    def get_log_level(self) -> int:
        r"""
        Returns the log Level used by this object.
        """
        return self._log.level

    @property
    def name(self):
        r"""
        Name of the module. This is relevant in some applications
        """
        if self._name is not None:
            return self._name
        raise AttributeError

    @name.setter
    def name(self, val: str):
        self._log.error("The name cannot be changed after initialization.")


_GLOBAL_LOGGER = _Logger(name="global")


def set_log_level(level: int):
    r"""
    Sets the logging Level of the toolbox. The best is to use a value of the logging module.
    For example

    .. code-block::
        import kerch
        import logging

        kerch.set_log_level(logging.DEBUG)


    .. note::
        Changing the log value does not affect the already instantiated objects.


    """
    _GLOBALS["LOG_LEVEL"] = level
    _GLOBAL_LOGGER.set_log_level(level)


def get_log_level() -> str:
    r"""
    Returns the logging Level of the toolbox.
    """
    return logging.getLevelName(_GLOBALS["LOG_LEVEL"])
