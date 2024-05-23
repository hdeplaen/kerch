# coding=utf-8
import torch
import logging

FTYPE = torch.float32
ITYPE = torch.int16

EPS = 1.e-7

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
    assert isinstance(type, torch.dtype), 'The type is not an instance of torch.dtype.'
    global FTYPE
    FTYPE = type
    logging.warning('Changing name has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')

def set_itype(type):
    assert isinstance(type, torch.dtype), 'The type is not an instance of torch.dtype.'
    global ITYPE
    ITYPE = type
    logging.warning('Changing name has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')


def set_eps(eps: float):
    assert eps>=0, 'The EPS value has to be positive'
    EPS = eps