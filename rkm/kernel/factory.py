def factory(kernel_type='linear', **kwargs):
    r"""
    Creates a kernel based on the specified type with the specified arguments. This is the same as
    calling `rkm.kernel.kernel_type(**kwargs)` (if `kernel_type` is not a string here). This allows for the creation of kernel where the type of
    kernel is passed as a string.

    :param kernel_type: Type of kernel chosen. For the possible choices, please refer to the (non-abstract) classes
        herebelow., defaults to `linear`
    :param \**kwargs: Arguments to be passed to the kernel constructor, such as `sample` or `sigma`. If an argument is
        passed that does not exist (e.g. `sigma` to a `linear` kernel), it will just be neglected. For the default
        values, please refer to the default values of the requested kernel.
    :type kernel_type: str, optional
    :type \**kwargs: dict, optional
    :return: An instantiation of the specified kernel.
    """
    from .linear import linear
    from .rbf import rbf
    from .hat import hat
    from .sigmoid import sigmoid
    from .indicator import indicator
    from .nystrom import nystrom
    from .polynomial import polynomial
    from .explicit_nn import explicit_nn
    from .implicit_nn import implicit_nn
    from .cosine import cosine

    switcher = {"linear": linear,
                "rbf": rbf,
                "explicit": explicit_nn,
                "implicit": implicit_nn,
                "polynomial": polynomial,
                "sigmoid": sigmoid,
                "indicator":  indicator,
                "hat": hat,
                "cosine": cosine,
                "nystrom": nystrom}
    if kernel_type not in switcher:
        raise NameError("Invalid kernel type.")
    return switcher[kernel_type](**kwargs)

