import torch

from ...feature import Sample
from .random_features import RandomFeatures


class RFArcsinh(RandomFeatures):
    def __init__(self, *args, **kwargs):
        super(RFArcsinh, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'random features arcsinh' + super(RFArcsinh, self).__str__()

    @property
    def hparams_fixed(self):
        return {"Kernel": "Random Features Arcsinh", **super(RandomFeatures, self).hparams_fixed}

    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x)

    def activation_fn_inv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x)
