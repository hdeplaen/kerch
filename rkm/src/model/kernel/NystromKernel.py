"""
File containing the feature kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch
import logging

import rkm.src
import rkm.src.utils as utils
import rkm.src.model.kernel.ExplicitKernel as ExplicitKernel
import rkm.src.model.kernel.KernelFactory as KernelFactory

class NystromKernel(ExplicitKernel.ExplicitKernel):
    """
    Nyström kernel
    """

    @rkm.src.kwargs_decorator({
        "dim": None,
        "base_type": "rbf",
        "base_centering": True
    })
    def __init__(self, **kwargs):
        """
        no specific parameters to the linear kernel
        """
        assert kwargs["base_type"] is not "nystrom", 'Cannot create a Nyström kernel based on another Nyström kernel.'

        super(NystromKernel, self).__init__(**kwargs)

        self._internal_kernel = KernelFactory.factory(**{**kwargs,
                                                         "centering": kwargs["base_centering"],
                                                         "kernel_type": kwargs["base_type"]})

        self.dim = kwargs["dim"]
        if self.dim is None:
            self.dim = self.init_kernels

        self.kernels_init(self.kernels)

    def __str__(self):
        return "Feature kernel"

    def hparams(self):
        return {"Kernel": "Feature", **super(NystromKernel, self).hparams}

    def kernels_init(self, x):
        super(NystromKernel, self).kernels_init(x)

        self._internal_kernel.kernels_init(x)
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

    def _explicit(self, x=None):
        if x is None:
            return self._sample_phi[self._idx_kernels,:]

        Ky = self._internal_kernel.k(x)
        return Ky @ self.H @ torch.diag(1/self.lambdas_sqrt)
