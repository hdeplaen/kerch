# coding=utf-8
from .single_view.Level import Level
def factory(level_type='kpca', **kwargs) -> Level:
    r"""
    Creates a kernel based on the specified name with the specified arguments. This is the same as
    calling `kerch.kernel.name(*args, **kwargs)` (if `name` is not a string here). This allows for the creation of kernel where
    the name of kernel is passed as a string.

    :param level_type: Type of level chosen. For the possible choices, please refer to the (non-abstract) classes
        herebelow., defaults to `kpca`
    :param \**kwargs: Arguments to be passed to the level constructor. If an argument is
        passed that does not exist, it will just be neglected. For the default
        values, please refer to the default values of the requested level.
    :type level_type: str, optional
    :type \**kwargs: dict, optional
    :return: An instantiation of the specified level.

    .. warning::
        Only supports single view levels.
    """

    if level_type.lower() not in ["kpca", "lssvm", "ridge"]:
        raise NameError(f"Invalid level type, currently only KPCA, LSSVM and Ridge supported.")

    def case_insensitive_getattr(obj, attr):
        for a in dir(obj):
            if a.lower() == attr.lower():
                return getattr(obj, a)
        return None

    import kerch.level.single_view
    level = case_insensitive_getattr(kerch.level.single_view, level_type)
    if level is None:
        raise NameError("Invalid level type.")

    return level(**kwargs)
