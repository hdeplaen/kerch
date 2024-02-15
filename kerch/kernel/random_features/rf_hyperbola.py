import torch

from ...feature import Sample
from .random_features import RandomFeatures


class RFHyperbola(RandomFeatures):
    r"""
    .. math::
        -\alpha x^2 + 2 \beta x y - \gamma y^2 + \delta = 0


    .. math::
        \sigma(x) = \frac{1}{\gamma}\left(\sqrt{(\beta^2-\alpha\gamma)x^2+\gamma\delta} + \beta x\right)

    .. math::
        \sigma^{-1}(x) = \frac{1}{\alpha}\left(\sqrt{(\beta^2-\alpha\gamma)y^2+\alpha\delta} + \beta y\right)

    """
    def __init__(self, *args, **kwargs):
        super(RFHyperbola, self).__init__(*args, **kwargs)
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', 1)
        self.delta = kwargs.pop('delta', 1)
        self.beta = kwargs.pop('beta', 2)


    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        assert val > 0, f"The value for alpha must be positive ({val:1.2f})."
        self._alpha = float(val)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, val: float):
        assert val > 0, f"The value for beta must be strictly positive ({val:1.2f})."
        assert val ** 2 > self.alpha * self.gamma, \
            (f"The value of beta square ({val**2:1.2f}) must be greater than alpha * gamma "
             f"({self.alpha:1.2f} * {self.gamma:1.2f} = {self.alpha * self.gamma:1.2f}).")
        self._beta = float(val)

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, val: float):
        assert val > 0, f"The value for gamma must be strictly positive ({val:1.2f})."
        self._gamma = float(val)

    @property
    def delta(self) -> float:
        return self._delta

    @delta.setter
    def delta(self, val: float):
        assert val > 0, f"The value for delta must be strictly positive ({val:1.2f})."
        self._delta = float(val)

    def __str__(self):
        return 'random features hyperbola' + super(RFHyperbola, self).__str__()

    @property
    def hparams_fixed(self):
        return {"Kernel": "Random Features Hyperbola", **super(RandomFeatures, self).hparams_fixed}

    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        fact = 1 / self.gamma
        term_1 = self.beta ** 2 - self.gamma * self.alpha
        term_2 = self.gamma * self.delta
        term_sqrt = term_1 * (x ** 2) + term_2
        return fact * (torch.sqrt(term_sqrt) + self.beta * x)

    def activation_fn_inv(self, x: torch.Tensor) -> torch.Tensor:
        fact = 1 / self.alpha
        term_1 = self.beta ** 2 - self.gamma * self.alpha
        term_2 = self.alpha * self.delta
        term_sqrt = term_1 * (x ** 2) + term_2
        return fact * (-torch.sqrt(term_sqrt) + self.beta * x)
