"""
Source code for the RKM toolbox.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from abc import ABCMeta, abstractmethod
import sys
from kerch._module._Logger import _Logger


class KerchError(Exception, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, cls=None, message=""):
        msg = message
        if msg == "" and hasattr(self, 'message'):
            msg = self.message

        if isinstance(cls, _Logger):
            msg = "[" + cls.__class__.__name__ + "] " + msg

        super(KerchError, self).__init__(msg)


class ImplicitError(KerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Implicit representation not available."
        super(ImplicitError, self).__init__(*args, **kwargs)


class ExplicitError(KerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Explicit representation not available.\n" \
                       "[Example 1]: The explicit representation does not exist as it lies in an infinite " \
                       "dimensional Hilbert space.\n" \
                       "[Example 2]: Only the inner product (implicit representation) is known, but not the " \
                       "original vectors."
        super(ExplicitError, self).__init__(*args, **kwargs)


class RepresentationError(KerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Unrecognized or unspecified representation (must be primal or dual)."
        super(RepresentationError, self).__init__(*args, **kwargs)


class BijectionError(KerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Mathematically undefined operation. A transform is not bijective, thus non invertible."
        super(BijectionError, self).__init__(*args, **kwargs)


class NotInitializedError(KerchError):
    def __init__(self, *args, **kwargs):
        self.message = "The model has not been initialized yet."
        super(NotInitializedError, self).__init__(*args, **kwargs)


class MultiViewError(KerchError):
    def __init__(self, *args, **kwargs):
        self.message = "Primal operations are not defined a multi-view context. You must ask them for the different " \
                       "known separately, if it exists for it."
        super(MultiViewError, self).__init__(*args, **kwargs)
