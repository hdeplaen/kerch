# coding=utf-8
import torch
import logging

FTYPE = torch.float32
ITYPE = torch.int32

def gpu_available() -> bool:
    r"""
    Returns whether GPU-enhanced computation is possible and configured on this machine.
    """
    import torch.cuda
    if torch.cuda.is_available():
        from ..feature.logger import _GLOBAL_LOGGER
        _GLOBAL_LOGGER._logger.info("Using CUDA version " + torch.version.cuda)
        return True
    return False

def set_ftype(type):
    r"""
    Sets the generic floating type :attr:`kerch.FTYPE` used throughout the package. Typical choices include half precision
    :attr:`torch.float16`, single precision :attr:`torch.float32` (default) and double precision :attr:`torch.float64`.

    :param type: Default floating type to be used.
    :type type: PyTorch type

    .. warning:
        This does not affect the already instantiated tensors. It is thus preferable to set this in the beginning of
        the code to avoid any type mismatch.
    """

    assert isinstance(type, torch.dtype), 'The type is not an instance of torch.dtype.'
    global FTYPE
    FTYPE = type
    logging.warning('Changing name has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')

def set_itype(type):
    r"""
    Sets the generic integer type :attr:`kerch.ITYPE` used throughout the package. Typical choices include short
    integers :attr:`torch.int16` (-32 768 to 32 767), classical integers :attr:`torch.int32` (-2^31-1 to
    2^31, default) and long integers :attr:`torch.int64` (-2^63-1 to 2^63). We do not advise on using unsigned integers
    because of their limited support in PyTorch.

    :param type: Default integer type to be used.
    :type type: PyTorch type

    .. warning:
        This does not affect the already instantiated tensors. It is thus preferable to set this in the beginning of
        the code to avoid any type mismatch.
    """
    assert isinstance(type, torch.dtype), 'The type is not an instance of torch.dtype.'
    global ITYPE
    ITYPE = type
    logging.warning('Changing name has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')


def set_eps(eps: float):
    r"""
    Sets the generic epsilon value used throughout the toolbox to guarantee stability.

    :param eps: Default epsilon type to be used.
    :type eps: float

    .. warning:
        It is preferable to set this in the beginning of the code to avoid any type mismatch, preferably after setting
        the data type.
    """
    assert eps>=0, 'The EPS value has to be positive'
    global EPS
    EPS = torch.tensor(eps, dtype=FTYPE)
    return EPS

EPS = set_eps(1.e-7)