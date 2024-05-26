# coding=utf-8
from functools import wraps


def kwargs_decorator(dict_kwargs):
    r"""
    Decorator that appends the dictionnary of method keyword arguments (kwargs) with additional values provided in
    `dict_kwargs`. If a keyword already exists in the original dictionnary, its value is neglected. This is useful for
    adding default values to the keyword arguments of methods, e.g., constructors. An alternative is to use the
    `kwargs.pop(keyword, default)` instead.
    
    :param dict_kwargs: Dictionnary to be appended to the original dictionary. Redundant keywords are neglected.
    :return: Appended dictionnary.

    :type dict_kwargs: dict
    :rtype: dict
    """
    def wrapper(f):
        @wraps(f)
        def inner_wrapper(*args, **kwargs):
            new_kwargs = {**dict_kwargs, **kwargs}
            return f(*args, **new_kwargs)

        return inner_wrapper

    return wrapper


class extend_docstring:
    r"""
    Decorator adding the documentation of the provided definition to the decorated definition. This useful for
    inheriting the documentation.
    """
    def __init__(self, method):
        self.doc = method.__doc__

    def __call__(self, function):
        if self.doc is not None:
            doc = function.__doc__
            function.__doc__ = self.doc
            if doc is not None:
                function.__doc__ = doc + function.__doc__
        return function
