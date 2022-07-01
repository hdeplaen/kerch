import torch
from .type import ITYPE, FTYPE
from .errors import RepresentationError

def castf(x, dev=None, tensor=True):
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

def casti(x, dev=None):
    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=False, dtype=ITYPE, device=dev)
    else:
        x = x.type(ITYPE)

    if dev is not None:
        x = x.to(dev)

    return x.squeeze()


def check_representation(representation: str = None, default: str = None, cls=None):
    if representation is None and default is not None:
        representation = default

    valid = ["primal", "dual"]
    if representation not in valid:
        raise RepresentationError(cls)
    return representation


def capitalize_only_first(val: str) -> str:
    return val[0].upper() + val[1:]
