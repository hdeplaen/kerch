# coding=utf-8
"""
This abstract class has only one purpose, add a self._log property meant to log various information across the toolbox.
Doing it this way allows to get the name of the class instantiated and print more precise messages.

Author: HENRI DE PLAEN
Date: June 2022
"""
from __future__ import annotations

import logging
from abc import ABCMeta
import sys

from .. import _GLOBALS


class Logger(object, metaclass=ABCMeta):
    r"""
    :param logging_level: Logging level for this specific instance.
        If the value is ``None``, the current default kerch global log level will be used.
        Defaults to ``None`` (default kerch logging level).
        We refer to the :doc:`/features/logger` documentation for further information.
    :type logging_level: int, optional
    """

    if hasattr(sys, 'gettrace') and sys.gettrace() is not None: # debugger is active
        _LOGGING_FORMAT = "Kerch %(levelname)s [%(name)s]: %(message)s [%(pathname)s:%(lineno)d]"
    else:
        _LOGGING_FORMAT = "Kerch %(levelname)s [%(name)s]: %(message)s"

    _kerch_format = logging.Formatter(_LOGGING_FORMAT)
    _kerch_handler = logging.StreamHandler()
    _kerch_handler.setFormatter(_kerch_format)

    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('logger_name', None)
        class_name = self.__class__.__name__
        if self._name is not None and class_name != "Logger":
            log_name = self._name + ' ' + class_name
        elif self._name is not None:
            log_name = self._name
        else:
            log_name = class_name

        self._logger_internal: logging.Logger = logging.getLogger(name=log_name)
        self._logger_internal.addHandler(Logger._kerch_handler)
        self.logging_level = kwargs.pop('logging_level', None)

        self._logger.debug('Instantiating')

    @property
    def _logger(self) -> logging.Logger:
        r"""
        Logger of the instance.

        Usage:

        .. exec_code::

            # --- hide: start ---
            import sys
            import logging
            logging.basicConfig(stream=sys.stdout)
            # --- hide: stop ---

            import kerch
            import logging

            class MyClass(kerch.feature.Logger):
                def __init__(self, *args, **kwargs):
                    super(MyClass, self).__init__(*args, **kwargs)
                    self._logger.info('Instantiation done information.')
                    self._logger.warn('Instantiation done warning.')

            print('First class with default logging level:')
            my_class1 = MyClass()

            print('\nSecond instance with logging.INFO logging level:')
            my_class2 = MyClass(logging_level=logging.INFO)

        """
        return self._logger_internal

    @property
    def logging_level(self) -> int:
        r"""
        Logging level of this specific instance.
        If the value is ``None``, the current default kerch global log Level will be used.
        Defaults to ``None`` (default global kerch level).
        We refer to the :doc:`/features/logger` documentation for further information.
        """
        return self._logger_internal.level

    @logging_level.setter
    def logging_level(self, level: int | None):
        if level is None:
            level = _GLOBALS["LOG_LEVEL"]
        self._logger_internal.setLevel(level)


_GLOBAL_LOGGER = Logger(logger_name="global")


def set_logging_level(level: int):
    r"""
    Changes the default logging level of the kerch package.

    :param level: Default kerch logging value
    :type level: int

    Usage:

    .. code-block:: python

        import kerch
        import logging

        kerch.set_logging_level(logging.DEBUG)


    .. warning::
        Changing the default logging value does not affect the already instantiated objects. We advise to set those values in
        the beginning of the code.


    """
    _GLOBALS["LOG_LEVEL"] = level
    _GLOBAL_LOGGER.logging_level = level


def get_logging_level() -> int:
    r"""
    Returns the default logging level of the kerch package.

    Usage:

    .. exec_code::

        import kerch
        import logging

        default_level = kerch.get_logging_level()
        default_level = logging.getLevelName(default_level)
        print(default_level)

    """
    return _GLOBALS["LOG_LEVEL"]
