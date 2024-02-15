import torch
from math import pi

from ...utils import extend_docstring, BijectionError
from ...feature import Sample
from .random_features import RandomFeatures

@extend_docstring(RandomFeatures)
class RFLReLU(RandomFeatures):
    def __init__(self, *args, **kwargs):
        super(RFLReLU, self).__init__(*args, **kwargs)
        self.alpha = kwargs.pop('alpha', .1)

    def __str__(self):
        return 'random Leaky-ReLU features' + super(RFLReLU, self).__str__()

    @property
    def hparams_fixed(self):
        return {"Kernel": "Random Leaky ReLU Features",
                "Kernel Negative Slope": self.alpha,
                **super(RandomFeatures, self).hparams_fixed}

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        self._reset_cache(avoid_classes=[Sample])
        self._alpha = val

    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x, negative_slope=self.alpha)

    def activation_fn_inv(self, x: torch.Tensor) -> torch.Tensor:
        try:
            fact = 1 / self.alpha
        except ZeroDivisionError:
            raise BijectionError(cls=self, message="Random Leaky-ReLU Features are not invertible for alpha=0.")
        if self.alpha <= 1:
            return torch.minimum(fact * x, x)
        return torch.maximum(fact * x, x)


    def closed_form_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        norm_y = torch.norm(y, p=2, dim=1, keepdim=True)
        x /= norm_x
        y /= norm_y
        u = x @ y.T
        K_unnorm = (1 + self.alpha ** 2) * u \
                   - ((1 - self.alpha) ** 2 / pi) * u * torch.acos(u) \
                   + ((1 - self.alpha) ** 2 / pi) * torch.sqrt(1 - u ** 2)
        return norm_x @ norm_y.T * K_unnorm / 2
