import torch

ftype = torch.float64
itype = torch.uint8

PLOT_ENV = None

from rkm.src.model import PrimalError, DualError, RepresentationError