from functools import wraps
import torch

ftype = torch.float32
itype = torch.uint8

PLOT_ENV = None

def kwargs_decorator(dict_kwargs):
    def wrapper(f):
        @wraps(f)
        def inner_wrapper(*args, **kwargs):
            new_kwargs = {**dict_kwargs, **kwargs}
            return f(*args, **new_kwargs)

        return inner_wrapper

    return wrapper