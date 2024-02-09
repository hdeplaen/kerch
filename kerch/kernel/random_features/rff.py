# coding=utf-8
"""
File containing the RFF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch
from math import sqrt, inf
from typing import Union

from ...utils import extend_docstring, FTYPE, castf
from .random_features import RandomFeatures
from ..generic.rbf import RBF


@extend_docstring(RandomFeatures)
class RFF(RandomFeatures):
    r"""
    Random Fourier Features kernel of :math:`\texttt{num_weights}` weights and (optional) bandwidth :math:`\sigma`.
    This can be seen as an explicit feature map approximation of :class:`..RBF`.

    .. math::
        \phi(x) = \frac{1}{\sqrt{\texttt{num_weights}}} \left(\begin{array}{cc}
            \cos(w_1^{\top}x / \sigma) \\
            \vdots \\
            \cos(w_{\texttt{num_weights}}^{\top}x / \sigma) \\
            \sin(w_1^{\top}x / \sigma) \\
            \vdots \\
            \sin(w_{\texttt{num_weights}}^{\top}x / \sigma) \\
        \end{array}\right)^\top

    with :math:`w_1, \ldots, w_{\texttt{num_weights}} \sim \mathcal{N}(0,I_{\texttt{dim_input}})` and :math:`\texttt{dim_feature} = 2 \times \texttt{num_weights}`.

    In the limit of :math:`\texttt{num_weights} \rightarrow +\infty`, we recover the RBF kernel:

    .. math::
        k(x,y) = \phi(x)^{\top}\phi(y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\sigma^2} \right)
    """

    def __init__(self, *args, **kwargs):
        super(RFF, self).__init__(*args, **kwargs)

    @property
    def dim_feature(self) -> int:
        r"""
        Dimension of the explicit feature map :math:`\texttt{dim_feature} = 2 \times \texttt{num_weights}`.
        """
        return 2 * self.num_weights

    @property
    def hparams_fixed(self):
        return {"Kernel": "Random Fourier Features",
                **super(RandomFeatures, self).hparams_fixed}

    def __str__(self):
        return "random fourier features" + super(RFF, self).__str__()

    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.cos(x), torch.sin(x)), dim=1)

    def closed_form_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._logger.info("For an infinite number of weights, you may consider implementing an RBF kernel directly.")
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        dist_square = torch.sum(diff * diff, dim=0, keepdim=False)
        return torch.exp(-0.5 * dist_square)