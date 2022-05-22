"""
File containing the feature kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch
import logging

from .. import utils
from .explicit import explicit, base
from .factory import factory

@utils.extend_docstring(base)
class nystrom(explicit):
    r"""
    Nyström kernel. Constructs an explicit feature map based on the eigendecomposition of any kernel matrix based on
    some sample.

    :param dim: Dimension of the explicit feature map to be constructed. This value cannot exceed the number of sample
        points. During eigendecomposition, very small eigenvalues are also going to pruned to avoid numerical
        instability. If `None`, the value will be assigned to `num_sample`., defaults to `None`
    :param base_type: The type of kernel on which the explicit feature map is going to be constructed., defaults to
        `"rbf"`
    :param base_centering: Specifies if the base kernel has to be centered., defaults to `True`
    :param \**kwargs: Other arguments for the base kernel (e.g. the bandwidth for an RBF kernel, the degree for a
        polynomial kernel etc.). For the default values, please refer to the requested class in question.
    :type dim: int, optional
    :type \**kwargs: dict, optional
    :type base_type: str, optional
    :type base_centering: bool, optional
    """

    @utils.kwargs_decorator({
        "dim": None,
        "base_type": "rbf",
        "base_centering": True
    })
    def __init__(self, **kwargs):
        assert kwargs["base_type"] != "nystrom", 'Cannot create a Nyström kernel based on another Nyström kernel.'

        super(nystrom, self).__init__(**kwargs)

        self._internal_kernel = factory(**{**kwargs,
                                            "centering": kwargs["base_centering"],
                                            "kernel_type": kwargs["base_type"]})

        self.dim = kwargs["dim"]
        if self.dim is None:
            self.dim = self._num_sample

        assert self.dim <= self._num_sample, 'Cannot construct an explicit feature map of greater dimension than the ' \
                                             'number of sample points.'

        self.init_sample(self._sample)

    def __str__(self):
        return "Feature kernel"

    def hparams(self):
        return {"Kernel": "Feature", **super(nystrom, self).hparams}

    def init_sample(self, sample=None):
        super(nystrom, self).init_sample(sample)

        self._internal_kernel.init_sample(self.sample)
        K = self._internal_kernel.K
        self.lambdas, self.H = utils.eigs(K, k=self.dim)

        # prune very small eigenvalues if they exist to avoid unstability due to the later inversion
        idx_small = self.lambdas < 1.e-10
        sum_small = torch.sum(idx_small)
        if sum_small > 0:
            logging.warning(
                f"Nyström kernel: {sum_small} very small eigenvalues are detected on {self.dim}."
                f"To avoid numerical instability, these values are pruned."
                f"The new explicit dimension is now {self.dim - sum_small}.")
            keep_idx = torch.logical_not(idx_small)
            self.lambdas = self.lambdas[keep_idx]
            self.H = self.H[:,keep_idx]
            self.dim -= sum_small

        self.lambdas_sqrt = torch.sqrt(self.lambdas)
        self._sample_phi = (self.H @ torch.diag(self.lambdas_sqrt)).data

    def update_sample(self, sample_values, idx_sample=None):
        raise NotImplementedError

    def _explicit(self, x=None):
        if x is None:
            return self._sample_phi[self._idx_sample, :]

        Ky = self._internal_kernel.k(x)
        return Ky @ self.H @ torch.diag(1/self.lambdas_sqrt)
