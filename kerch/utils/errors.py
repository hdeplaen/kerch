# coding=utf-8
from abc import ABCMeta, abstractmethod
from ..feature.logger import Logger
from .decorators import extend_docstring

class KerchError(Exception, metaclass=ABCMeta):
    r"""
    :param cls: Optional class throwing the exception. This is helpful to add a context to the error message.
    :param message: Optional error message.

    :type cls: Instance of :class:`kerch.feature.Logger`
    :type message: str
    """
    @abstractmethod
    def __init__(self, cls=None, message=""):
        msg = message
        if msg == "" and hasattr(self, 'message'):
            msg = self.message

        if isinstance(cls, Logger):
            msg = "[" + cls.__class__.__name__ + "] " + msg

        super(KerchError, self).__init__(msg)


@extend_docstring(KerchError)
class ImplicitError(KerchError):
    r"""
    Error thrown when an implicit representation is requested and mathematically not available.
    """
    def __init__(self, *args, **kwargs):
        self.message = "Implicit representation not available."
        super(ImplicitError, self).__init__(*args, **kwargs)


@extend_docstring(KerchError)
class ExplicitError(KerchError):
    r"""
    Error thrown when an explicit representation is requested and mathematically not available.
    """
    def __init__(self, *args, **kwargs):
        self.message = "Explicit representation not available.\n" \
                       "[Example 1]: The explicit representation does not exist as it lies in an infinite " \
                       "dimensional Hilbert space.\n" \
                       "[Example 2]: Only the inner product (implicit representation) is known, but not the " \
                       "original vectors."
        super(ExplicitError, self).__init__(*args, **kwargs)


@extend_docstring(KerchError)
class RepresentationError(KerchError):
    r"""
    Error thrown when the requested representation is invalid.
    """
    def __init__(self, *args, **kwargs):
        self.message = "Unrecognized or unspecified representation (must be primal or dual)."
        super(RepresentationError, self).__init__(*args, **kwargs)


@extend_docstring(KerchError)
class BijectionError(KerchError):
    r"""
    Error thrown when an inverse is requested from a non-bijective function.
    """
    def __init__(self, *args, **kwargs):
        self.message = "Mathematically undefined operation. A transform is not bijective, thus non invertible."
        super(BijectionError, self).__init__(*args, **kwargs)


@extend_docstring(KerchError)
class NotInitializedError(KerchError):
    r"""
    Error thrown when a model is asked to perform some tasks, but is not yet fully initialized.
    """
    def __init__(self, *args, **kwargs):
        self.message = "The model has not been initialized yet."
        super(NotInitializedError, self).__init__(*args, **kwargs)


@extend_docstring(KerchError)
class MultiViewError(KerchError):
    r"""
    Error thrown when some operations are requested, but not defined in a multi-view context.
    """
    def __init__(self, *args, **kwargs):
        self.message = "Primal operations are not defined a multi-view context. You must ask them for the different " \
                       "known separately, if it exists for it."
        super(MultiViewError, self).__init__(*args, **kwargs)
