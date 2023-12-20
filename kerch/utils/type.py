import torch
import logging

FTYPE = torch.float32
ITYPE = torch.int16

EPS = 1.e-10

def set_ftype(type):
    FTYPE = type
    logging.warning('Changing name has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')

def set_itype(type):
    ITYPE = type
    logging.warning('Changing name has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')