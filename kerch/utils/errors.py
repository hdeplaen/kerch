"""
Source code for the RKM toolbox.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from abc import ABCMeta, abstractmethod
from .._Logger import _Logger


class kerchError(Exception, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, cls=None):

        msg = ""
        if hasattr(self, 'message'):
            msg = self.message

        if isinstance(cls, _Logger):
            cls._log.error(msg)
            msg = "[" + cls.__class__.__name__ + "] " + msg

        super(kerchError, self).__init__(msg)


class ImplicitError(kerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Implicit representation not available."
        super(ImplicitError, self).__init__(*args, **kwargs)


class ExplicitError(kerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Explicit representation not available.\n" \
                       "[Example 1]: The explicit representation does not exist as it lies in an infinite " \
                       "dimensional Hilbert space.\n" \
                       "[Example 2]: Only the inner product (implicit representation) is known, but not the " \
                       "original vectors."
        super(ExplicitError, self).__init__(*args, **kwargs)


class RepresentationError(kerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Unrecognized or unspecified representation (must be primal or dual)."
        super(RepresentationError, self).__init__(*args, **kwargs)

class BijectionError(kerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Mathematically undefined operation. A projection is not bijective, thus non invertible."
        super(BijectionError, self).__init__(*args, **kwargs)


class MultiViewError(kerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Primal operations are not defined a multi-view context. You must ask them for the different " \
                       "known separately, if it exists for it."
        super(MultiViewError, self).__init__(*args, **kwargs)
