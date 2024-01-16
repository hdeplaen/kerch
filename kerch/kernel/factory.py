# coding=utf-8
from .Kernel import Kernel
from ..utils import DEFAULT_KERNEL_TYPE


def factory(kernel_type=DEFAULT_KERNEL_TYPE, **kwargs) -> Kernel:
    r"""
    Creates a kernel based on the specified name with the specified arguments. This is the same as
    calling `kerch.kernel.name(*args, **kwargs)` (if `name` is not a string here). This allows for the creation of kernel where
    the name of kernel is passed as a string.

    :param kernel_type: Type of kernel chosen. For the possible choices, please refer to the (non-abstract) classes
        herebelow. Defaults to kerch.DEFAULT_KERNEL_TYPE.
    :param \**kwargs: Arguments to be passed to the kernel constructor, such as `sample` or `sigma`. If an argument is
        passed that does not exist (e.g. `sigma` to a `linear` kernel), it will just be neglected. For the default
        values, please refer to the default values of the requested kernel.
    :type kernel_type: str, optional
    :type \**kwargs: dict, optional
    :return: An instantiation of the specified kernel.
    """

    kernel = class_factory(kernel_type)
    return kernel(**kwargs)


def class_factory(kernel_type=DEFAULT_KERNEL_TYPE):
    def case_insensitive_getattr(obj, attr):
        for a in dir(obj):
            if a.lower() == attr.lower():
                return getattr(obj, a)
        return None

    import kerch.kernel
    kernel = case_insensitive_getattr(kerch.kernel, kernel_type.replace("_", ""))
    if kernel is None:
        raise NameError("Invalid kernel type.")
    return kernel

