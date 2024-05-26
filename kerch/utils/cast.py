# coding=utf-8
from __future__ import annotations
import torch
from .type import ITYPE, FTYPE
from .errors import RepresentationError

def castf(x, dev=None, tensor=True) -> torch.Tensor | None:
    r"""
    Casts the input to a PyTorch float tensor. If the input is a scalar, it is cast to a tensor. The cast can be done
    to a 1D or a 2D tensor depending on the parameter `tensor`. If the provided data x has more than 2 dimensions, an
    error is raised. The default floating type used is :attr:`kerch.FTYPE`. The `None` values are not casted and 
    returned as is.

    :param x: The input to cast.
    :param dev: The device to cast the tensor to. Defaults to `None`, which corresponds to no device change.
    :param tensor: If True, the input is cast to a 2D tensor. If False, the input is cast to a 1D tensor.
    :return: The input cast to a PyTorch float tensor, with optional device.

    :type x: float | torch.Tensor | np.ndarray | None
    :type dev: Optional[torch.device]
    :type tensor: bool
    :rtype: torch.Tensor | None
    """
    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=False, dtype=FTYPE, device=dev)
    else:
        x = x.type(FTYPE)

    if dev is not None:
        x = x.to(dev)

    if tensor:
        dim = len(x.shape)
        if dim == 0:
            x = x.unsqueeze(0)
            dim = 1
        if dim == 1:
            x = x.unsqueeze(1)
        elif dim > 2:
            raise NameError(f"Provided data has too much dimensions ({dim}).")

    return x

def casti(x, dev=None, tensor=False) -> torch.Tensor | None:
    r"""
    Casts the input to a PyTorch integer tensor. If the input is a scalar, it is cast to a tensor. The cast can be done
    to a 1D or a 2D tensor depending on the parameter `tensor`. If the provided data x has more than 2 dimensions, an
    error is raised. The default floating type used is :attr:`kerch.ITYPE`. The `None` values are not casted and 
    returned as is.

    :param x: The input to cast.
    :param dev: The device to cast the tensor to. Defaults to `None`, which corresponds to no device change.
    :param tensor: If True, the input is cast to a 2D tensor. If False, the input is cast to a 1D tensor.
    :return: The input cast to a PyTorch integer tensor, with optional device.

    :type x: int | torch.Tensor | np.ndarray | None
    :type dev: Optional[torch.device]
    :type tensor: bool
    :rtype: torch.Tensor | None
    """
    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=False, dtype=ITYPE, device=dev)
    else:
        x = x.type(ITYPE)

    if dev is not None:
        x = x.to(dev)

    if tensor:
        dim = len(x.shape)
        if dim == 0:
            x = x.unsqueeze(0)
            dim = 1
        if dim == 1:
            x = x.unsqueeze(1)
        elif dim > 2:
            raise NameError(f"Provided data has too much dimensions ({dim}).")

    return x.squeeze()


def check_representation(representation: str = None, default: str = None, cls=None) -> str:
    r"""
    This model verifies whether the provided representation is valid. If the representation is `None` and a default
    value is provided, the default value is returned. If the representation is not `None` and is not valid, an error is
    raised. The valid representations are `primal` and `dual`.
    
    :param representation: The representation to check.
    :param default: Default representation for the case where `representation` is `None`.
    :param cls: An instance of :class:`kerch.feature.Logger` to throw the error from, typically the one calling this
        method. This is optional.
    :return: "primal" | "dual"

    :type representation: str, optional
    :type default: str, optional
    :type cls: kerch.feature.Logger, optional
    :rtype: str
    """
    if representation is None and default is not None:
        representation = default

    valid = ["primal", "dual"]
    if representation not in valid:
        raise RepresentationError(cls)
    return representation


def capitalize_only_first(val: str) -> str:
    r"""
    This method returns the input string with the first letter capitalized and the rest of the string unchanged.

    :param val: String to be capitalized.
    :return: Capitalized string.
    :type val: str
    :rtype: str
    """
    return val[0].upper() + val[1:]
