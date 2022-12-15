from ._statistics import _Statistics


def factory(type='linear', **kwargs) -> _Statistics:
    r"""
    Creates a kernel based on the specified type with the specified arguments. This is the same as
    calling `kerpy.kernel.type(**kwargs)` (if `type` is not a string here). This allows for the creation of kernel where
    the type of kernel is passed as a string.

    :param type: Type of kernel chosen. For the possible choices, please refer to the (non-abstract) classes
        herebelow., defaults to `linear`
    :param \**kwargs: Arguments to be passed to the kernel constructor, such as `sample` or `sigma`. If an argument is
        passed that does not exist (e.g. `sigma` to a `linear` kernel), it will just be neglected. For the default
        values, please refer to the default values of the requested kernel.
    :type type: str, optional
    :type \**kwargs: dict, optional
    :return: An instantiation of the specified kernel.
    """

    def case_insensitive_getattr(obj, attr):
        for a in dir(obj):
            if a.lower() == attr.lower():
                return getattr(obj, a)
        return None

    import kerch.kernel
    kernel = case_insensitive_getattr(kerch.kernel, type.replace("_", ""))
    if kernel is None:
        raise NameError("Invalid kernel type.")

    return kernel(**kwargs)
