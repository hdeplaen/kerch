import torch
from .type import ITYPE, FTYPE

def castf(x, tensor=True):
    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=False, dtype=FTYPE)
    else:
        x = x.type(FTYPE)

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

def casti(x):
    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=False, dtype=ITYPE)
    else:
        x = x.type(ITYPE)

    return x.squeeze()