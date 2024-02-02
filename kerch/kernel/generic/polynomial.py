# coding=utf-8
"""
File containing the polynomial kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from __future__ import annotations
from typing import Iterator
from math import factorial, comb, prod, sqrt, inf

from torch import Tensor

from ... import utils
from ..kernel import Kernel
from ..explicit import Explicit
from ..implicit import Implicit

import torch


@utils.extend_docstring(Kernel)
class Polynomial(Implicit, Explicit):
    r"""
    Polynomial kernel of degree :math:`\alpha \geq 0` and parameter :math:`\beta`.

    .. math::
        k(x,y) = \left(x^\top y + \beta\right)^\alpha.

    Provided the degree :math:`\alpha` is a natural number, this kernel accepts both an explicit feature map and an equivalent kernel formulation not depending on the
    inner product of the explicit feature maps (implicit). Its components are given by

    .. math::
        \left[\phi(x)\right]_k = \sqrt{\frac{\alpha!}{j_0!j_1! \ldots j_\texttt{dim_input}!}}x_0^{j_0}x_1^{j_1}\ldots x_{\texttt{dim_input}-1}^{j_{\texttt{dim_input}-1}}\sqrt{\beta}^{j_\texttt{dim_input}},


    where :math:`k = 0, \ldots, \texttt{dim_feature}-1` correspond to all permutations satisfying

    .. math::
        j_0 + j_1 + \ldots + j_{\texttt{dim_input}} = \alpha,

    and

    .. math::
            \texttt{dim_feature} = \left(\begin{array}{c}\texttt{dim_input} + \alpha \\ \alpha\end{array}\right).

    One can verify that :math:`k(x,y) = \phi(x)^\top\phi(y)`. An example is also given in the Example section of its
    documentation.

    .. note::
        For a natural number degree :math:`\alpha` . The computation of a kernel matrix of :math:`n` points is typically
        more efficient to compute as an inner product of the explicit feature map if :math:`\texttt{dim_feature} < n`
        and using the kernel formula otherwise. If the degree is not a natural number, only the latter is possible as
        the explicit feature map does not exist.

        This can be specified when calling :py:meth:`..Polynomial.k` by specifying the boolean ``explicit`` to
        ``True`` (using the explicit feature map) or ``False`` (directly using the kernel formula).

        Other considerations may come into play. If a centered or normalized kernel on an out-of-sample is required, this may require extra
        computations when directly using the kernel matrix as doing it on the explicit feature is more straightforward.

    :param alpha: Degree :math:`\alpha` of the polynomial kernel. Defaults to 2.
    :param beta: Value :math:`\beta` of the polynomial kernel. Defaults to 1.
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
        r"""
        True if the method has an explicit formulation, False otherwise. This is the case only if the degree
        :math:`\alpha` is a natural number.
        """
        if (self.alpha % 1) == 0:
            return True
        return False

    @property
    def dim_feature(self) -> int | inf:
        r"""
        Feature dimension. Provided the degree :math:`\alpha` is a natural number, it is given by

        .. math::
            \texttt{dim_feature} = \left(\begin{array}{c}\texttt{dim_input} + \alpha \\ \alpha\end{array}\right).


        If the degree :math:`\alpha` is not a natural number, the explicit feature does not exist and by consequence
        the feature dimension is infinite.
        """
        if (self.alpha % 1) == 0:
            alpha = int(self.alpha)
            return comb(self.dim_input + alpha, alpha)
        else:
            return inf

    @property
    def alpha(self):
        r"""
        Degree of the polynomial. This is argument plays a similar role to the bandwidth of
        an exponential kernel, such as the RBF kernel.

        .. note::
            The explicit feature map only exists if the degree is a natural number.

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
        Boolean indicating if the alpha/degree is trainable. In other words, this argument provides or not a gradient
        to the degree for potential gradient-based training.
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
    def hparams_variable(self):
        return {'Kernel alpha': self.alpha,
                'Kernel beta': self.beta,
                **super(Polynomial, self).hparams_variable}

    @property
    def hparams_fixed(self):
        return {"Kernel": "Polynomial"}

    @property
    def _phi_fun(self):
        if (self.alpha % 1) != 0:
            raise utils.ExplicitError(cls=self,
                                      message='Explicit formulation is only possible for degrees that are natural numbers.')

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

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield self._alpha
        yield self._beta
        yield from super(Polynomial, self)._slow_parameters(recurse)

    @utils.extend_docstring(Kernel.k)
    def k(self, x=None, y=None, explicit=None, transform=None) -> torch.Tensor:
        r"""
        .. note::
            For the specific case of the polynomial kernel, the optimal value for ``explicit`` is automatically determined
            based on the size of the inputs if ``explicit=None``. This does not take into account possible transforms.
        """

        # automatically determining best representation
        x = utils.castf(x)
        y = utils.castf(x)
        if explicit is None:
            if x is None:
                num_x = self.num_idx
            else:
                num_x = x.shape[0]
            if y is None:
                num_y = self.num_idx
            else:
                num_y = y.shape[0]
            if self.dim_feature * (num_x + num_y) < num_x * num_y:
                explicit = True
            else:
                explicit = False

        return super(Polynomial, self).k(x=x, y=y, explicit=explicit, transform=transform)

    @utils.extend_docstring(Kernel.explicit_preimage)
    def explicit_preimage(self, phi: Tensor):
        return NotImplementedError