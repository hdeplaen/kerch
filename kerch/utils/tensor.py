from typing import Union

import torch
from torch import Tensor as T


def eye_like(m: T) -> T:
    return torch.eye(*m.size(), out=torch.empty_like(m))


def ones_like(m: T) -> T:
    return torch.ones(*m.size(), out=torch.empty_like(m))


def equal(val1: Union[T, None], val2: Union[T, None]):
    if isinstance(val1, T) and isinstance(val2, T):
        return torch.equal(val1, val2)
    elif isinstance(val1, str) and isinstance(val2, str):
        return val1 == val2
    elif val1 is None and val2 is None:
        return True
    else:
        return False
