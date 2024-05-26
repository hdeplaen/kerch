# coding=utf-8
from typing import Union

import torch
from torch import Tensor as T


def eye_like(m: T) -> T:
    r"""
    Creates an identity matrix of the same size, data type and on the same device as the provided one.
    :param m: Provided matrix to create an identity matrix from.
    :return: Identity matrix of the same size, data type and on the same device.

    :type m: torch.Tensor
    :rtype: torch.Tensor
    """
    return torch.eye(*m.size(), out=torch.empty_like(m))


def ones_like(m: T) -> T:
    r"""
    Creates a matrix full of ones of the same size, data type and on the same device as the provided one.
    :param m: Provided matrix to create the new matrix from.
    :return: Matrix full of ones matrix of the same size, data type and on the same device.

    :type m: torch.Tensor
    :rtype: torch.Tensor
    """
    return torch.ones(*m.size(), out=torch.empty_like(m))


def equal(val1: Union[T, None], val2: Union[T, None]) -> bool:
    r"""
    Verifies whether the two provided tensors are identical in content. Can also provide `None` values which will only
    be identical to other `None` values.

    :param val1: First tensor to compare.
    :param val2: Second tensor to compare.
    :return: `True` if the two provided tensors are identical, `False` otherwise.

    :type val1: Union[T, None]
    :type val2: Union[T, None]
    :rtype: bool
    """
    if isinstance(val1, T) and isinstance(val2, T):
        return torch.equal(val1, val2)
    elif isinstance(val1, str) and isinstance(val2, str):
        return val1 == val2
    elif val1 is None and val2 is None:
        return True
    else:
        return False
