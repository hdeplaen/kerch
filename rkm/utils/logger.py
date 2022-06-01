import logging

_LOGGING_FORMAT = "%(name)s %(levelname)s [%(module)s]: %(message)s"
# _LOGGING_FORMAT = "%(name)s %(levelname)s: %(message)s\n\t[Source: %(pathname)s]"
_kerpy_format = logging.Formatter(_LOGGING_FORMAT)
_kerpy_handler = logging.StreamHandler()
_kerpy_handler.setFormatter(_kerpy_format)

logger = logging.getLogger(name="KerPy")
logger.addHandler(_kerpy_handler)
