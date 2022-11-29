import torch
import logging

FTYPE = torch.float64
ITYPE = torch.int16

def set_ftype(type):
    FTYPE = type
    logging.warning('Changing type has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')

def set_itype(type):
    ITYPE = type
    logging.warning('Changing type has to be carefully considered as changes '
                    'after initialization may lead to inconsistencies.')