import logging
from .. import _LOG_LEVEL


class _logger(object):
    def __init__(self, *args, **kwargs):
        _LOGGING_FORMAT = "KerPy %(levelname)s [%(name)s]: %(message)s"

        _kerpy_format = logging.Formatter(_LOGGING_FORMAT)
        _kerpy_handler = logging.StreamHandler()
        _kerpy_handler.setFormatter(_kerpy_format)

        self._log = logging.getLogger(name=self.__class__.__name__)
        self._log.addHandler(_kerpy_handler)
        self._log.setLevel(_LOG_LEVEL)


def set_log_level(level: int):
    _LOG_LEVEL = level


def get_log_level() -> int:
    return _LOG_LEVEL
