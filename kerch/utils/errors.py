"""
Source code for the RKM toolbox.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from abc import ABCMeta, abstractmethod
from .._logger import _Logger


class KerPyError(Exception, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, cls=None):

        msg = ""
        if hasattr(self, 'message'):
            msg = self.message

        if isinstance(cls, _Logger):
            cls._log.error(msg)
            msg = "[" + cls.__class__.__name__ + "] " + msg

        super(KerPyError, self).__init__(msg)


class DualError(KerPyError):
    def __init__(self, *args, **kwargs):
        self.message = "Dual representation not available."
        super(DualError, self).__init__(*args, **kwargs)


class PrimalError(KerPyError):
    def __init__(self, *args, **kwargs):
        self.message = "Primal representation not available. " \
                       "The explicit representation lies in an infinite dimensional Hilbert space."
        super(PrimalError, self).__init__(*args, **kwargs)


class RepresentationError(KerPyError):
    def __init__(self, *args, **kwargs):
        self.message = "Unrecognized or unspecified representation (must be primal or dual)."
        super(RepresentationError, self).__init__(*args, **kwargs)


class MultiViewError(KerPyError):
    def __init__(self, *args, **kwargs):
        self.message = "Primal operations are not defined a multi-view context. You must ask them for the different " \
                       "views separately, if it exists for it."
        super(MultiViewError, self).__init__(*args, **kwargs)
