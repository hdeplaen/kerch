# coding=utf-8
from .Model import Model

def factory(model_type:str = 'rkm', **kwargs) -> Model:
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

    model = class_factory(model_type)
    return model(**kwargs)

def class_factory(model_type: str = 'rkm'):
    def case_insensitive_getattr(obj, attr):
        for a in dir(obj):
            if a.lower() == attr.lower():
                return getattr(obj, a)
        return None

    import kerch.model
    model = case_insensitive_getattr(kerch.model, model_type)
    if model is None:
        raise NameError("Invalid model type.")
    return model

