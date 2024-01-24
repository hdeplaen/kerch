# coding=utf-8
"""
File containing the polynomial kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from typing import Iterator
from math import factorial, comb, prod, sqrt

from .. import utils
from .kernel import Kernel

import torch


@utils.extend_docstring(Kernel)
class Polynomial(Kernel):
    r"""
    Polynomial kernel.

    .. math::
        k(x,y) = \left(x^\top y + \beta\right)^\alpha.

    .. note ::
        An explicit feature map also corresponds to this kernel, but is not implemented.

    :param alpha: Degree of the polynomial kernel. Defaults to 2
    :param beta: Value beta of the polynomial kernel. Defaults to 1
    :param alpha_trainable: `True` if the gradient of the degree is to be computed. If so, a graph is computed
        and the degree can be updated. `False` just leads to a static computation., defaults to `False`
    :param beta_trainable: `True` if the gradient of the degree is to be computed. If so, a graph is computed
        and the degree can be updated. `False` just leads to a static computation., defaults to `False`
    :type alpha: double, optional
    :type beta: double, optional
    :type alpha_trainable: bool, optional
    :type beta_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"alpha": 2., "alpha_trainable": False,
         "beta": 1., "beta_trainable": False})
    def __init__(self, *args, **kwargs):
        self._alpha = kwargs.pop("alpha", 2.)
        self._beta = kwargs.pop('beta', 1.)
        super(Polynomial, self).__init__(*args, **kwargs)

        self._alpha_trainable = kwargs.pop("alpha_trainable", False)
        self._beta_trainable = kwargs.pop("beta_trainable", False)
        self._alpha = torch.nn.Parameter(torch.tensor(self._alpha),
                                         requires_grad=self.alpha_trainable)
        self._beta = torch.nn.Parameter(torch.tensor(self._beta),
                                        requires_grad=self.beta_trainable)

    def __str__(self):
        return f"polynomial kernel (alpha={self.alpha}, beta={self.beta})"

    @property
    def explicit(self) -> bool:
        self._log.debug("The explicit formulation of the polynomial kernel is not implemented.")
        return False

    @property
    def dim_feature(self) -> int:
        assert (self.alpha % 1) == 0, 'Explicit formulation is only possible for degrees that are natural numbers.'
        alpha = int(self.alpha)
        return comb(self.dim_input + alpha, alpha)

    @property
    def alpha(self):
        r"""
        Degree of the polynomial.

        .. note::
            The explicit feature map only exists if the degree is a finite natural number.

        """
        if isinstance(self._alpha, torch.nn.Parameter):
            return self._alpha.detach().cpu().numpy()
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._reset_cache(reset_persisting=False)
        self._remove_from_cache("_poly_explicit")
        self._alpha.data = val

    @property
    def alpha_trainable(self) -> bool:
        r"""
        Boolean indicating if the alpha/degree is trainable. This is argument plays a similar role to the bandwidth of
        an exponential kernel, such as the RBF kernel.
        """
        return self._alpha_trainable

    @property
    def beta(self):
        r"""
        Beta of the polynomial.
        """
        if isinstance(self._beta, torch.nn.Parameter):
            return self._beta.detach().cpu().numpy().item()
        return self._beta

    @alpha.setter
    def alpha(self, val):
        self._reset_cache(reset_persisting=False)
        self._remove_from_cache("_poly_explicit")
        self._beta.data = val

    @property
    def beta_trainable(self) -> bool:
        r"""
        Boolean indicating if the beta is trainable.
        """
        return self._beta_trainable

    @property
    def params(self):
        return {'Kernel alpha': self.alpha,
                'Kernel beta': self.beta,
                **super(Polynomial, self).params}

    @property
    def hparams(self):
        return {"Kernel": "Polynomial"}

    @property
    def _phi_fun(self):
        assert (self.alpha % 1) == 0, 'Explicit formulation is only possible for degrees that are natural numbers.'

        def multichoose(n, k):
            if not k: return [[0] * n]
            if not n: return []
            if n == 1: return [[k]]
            return [[0] + val for val in multichoose(n - 1, k)] + \
                [[val[0] + 1] + val[1:] for val in multichoose(n, k - 1)]

        permutations = multichoose(self.dim_input + 1, int(self.alpha))

        sqrt_d_fact = sqrt(factorial(int(self.alpha)))
        sqrt_beta = sqrt(self.beta)

        permutations = utils.casti(permutations)

        def permutation_out(x: torch.Tensor, perm):
            denominator = sqrt(prod(map(factorial, perm)))
            fact = sqrt_d_fact / denominator
            return fact * torch.prod(x.pow((perm[0:-1])[None, :]), dim=1, keepdim=False) * (sqrt_beta ** perm[-1])

        def phi(x):
            sol = torch.zeros((x.shape[0], self.dim_feature))
            for i, perm in enumerate(permutations):
                sol[:, i] = permutation_out(x, perm)
            return sol

        return self._get("_poly_explicit_fun", fun=lambda: phi, level_key='_poly_explicit_fun',
                         persisting=True, overwrite=self.alpha_trainable)

    def _implicit(self, x, y):
        return (x @ y.T + self._beta) ** self._alpha

    def _explicit(self, x):
        assert (self.alpha % 1) == 0, 'Explicit formulation is only possible for degrees that are natural numbers.'
        return self._phi_fun(x)

    def explicit_preimage(self, phi):
        raise NotImplementedError

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield self._alpha
        yield self._beta
        yield from super(Polynomial, self)._slow_parameters(recurse)


