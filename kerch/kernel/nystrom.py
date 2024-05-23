# coding=utf-8
"""
File containing the feature kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch

from .. import utils
from .explicit import Explicit, Kernel
from ._base_kernel import _BaseKernel
from ._factory import factory

@utils.extend_docstring(Kernel)
class Nystrom(Explicit):
    r"""
    Nyström kernel. Constructs an explicit feature map based on the eigendecomposition of any kernel matrix based on
    some sample.

    :param dim: Dimension of the explicit feature map to be constructed. This value cannot exceed the number of sample
        points. During eigendecomposition, very small eigenvalues are also going to be pruned to avoid numerical
        instability. If `None`, the value will be assigned to `num_sample`., defaults to `None`
    :param base_kernel_type: The name of kernel on which the explicit feature map is going to be constructed. Default to
        kerch.DEFAULT_KERNEL_TYPE
    :param base_kernel_transform: Same as kernel_transform but for the base kernel, when using the factory through
        `base_type`. Defaults to [].
    :param \**kwargs: Other arguments for the _Projected kernel (e.g. the bandwidth for an RBF kernel, the degree for a
        polynomial kernel etc.). For the default values, please refer to the requested class in question.
    :param base_kernel: Instead of creating a new kernel on which to use the Nyström method, one can also perform it
        on an existing kernel. In that case, the other _Projected arguments are bypassed., defaults to `None`
    :type dim: int, optional
    :type \**kwargs: dict, optional
    :type base_type: str, optional
    :type base_kernel_transforms: list(str), optional
    :type base_kernel: kerch.kernel.*, optional
    """

    def __init__(self, *args, **kwargs):
        base_kernel_type = kwargs.pop('base_kernel_type', utils.DEFAULT_KERNEL_TYPE)
        assert base_kernel_type.lower() != "nystrom", 'Cannot create a Nyström kernel based on another Nyström ' \
                                                         'kernel.'
        self._base_kernel = None

        k = kwargs.pop('base_kernel', None)
        assert not isinstance(k, str), "base_kernel must be of kernel type (use base_type instead)."
        if k is None:
            # normal case with a kernel created from the factory
            super(Nystrom, self).__init__(*args, **kwargs)

            self._base_kernel = factory(**{**kwargs,
                                               "kernel_type": base_kernel_type,
                                               "kernel_transform": kwargs.pop('base_kernel_transform', [])})
            self._base_kernel.init_sample(sample=self.current_sample_projected, idx_sample=self.idx)
        else:
            # nystromizing some existing kernel
            assert isinstance(k, _BaseKernel), "The provided kernel is not of the kernel class."
            super(Nystrom, self).__init__(**{**kwargs,
                                             "sample": k.sample,
                                             "sample_trainable": k.sample_trainable,
                                             "idx_sample": k.idx})
            self._base_kernel = k
            self._logger.info("Keeping original kernel transform (no overwriting, so base_kernel_transform is "
                           "neglected).")

        self._dim = kwargs.pop('dim', None)
        if self._dim is None:
            self.dim = self._num_total

    @property
    def dim(self):
        """
        Dimension of the explicit feature map.
        """
        return self._dim.cpu().numpy().item(0)

    @dim.setter
    def dim(self, val):
        if self._num_total is not None:
            assert val <= self._num_total, 'Cannot construct an explicit feature map of greater dimension than ' \
                                           'the number of sample points.'
        self._dim = utils.casti(val)
        self._reset_cache(reset_persisting=False)

    @property
    def dim_feature(self) -> int:
        self._compute_decomposition()
        return self.dim

    def __str__(self):
        return "Nystrom kernel"

    @property
    def base_kernel(self):
        r"""
            Kernel on which Nystrom performs the decomposition.
        """
        if self._base_kernel is None:
            raise utils.NotInitializedError(cls=self, message='The base kernel has not been defined yet.')
        return self._base_kernel

    def hparams_fixed(self):
        return {"Kernel": "Nystrom",
                "Base Kernel": self._base_kernel.hparams_fixed['Kernel'],
                **super(Nystrom, self).hparams_fixed}

    def init_sample(self, sample=None, idx_sample=None, prop_sample=None):
        super(Nystrom, self).init_sample(sample=sample, idx_sample=idx_sample, prop_sample=prop_sample)
        if self._base_kernel is not None:
            self._base_kernel.init_sample(sample=self.current_sample_projected, idx_sample=self.idx)

    @torch.no_grad()
    def _compute_decomposition(self):
        if "H" not in self.cache_keys(private=True):
            self._logger.info("Computing the eigendecomposition for the Nystrom kernel.")

            if self._dim is None:
                self.dim = self._num_total

            K = self._base_kernel.K
            lambdas, H = utils.eigs(K, k=self._dim)

            # verify that the decomposed kernel is PSD
            sum_neg = torch.sum(lambdas < 0)
            if sum_neg > 0:
                self._logger.warning(f"The decomposed kernel is not positive semi-definite as it possesses {sum_neg} "
                                  f"negative eigenvalues. These will be discarded, but may prove relevant if their "
                                  f"magnitude is non-negligible.")

            # prune very small eigenvalues if they exist to avoid unstability due to the later inversion
            idx_small = lambdas < utils.EPS
            sum_small = torch.sum(idx_small)
            if sum_small > 0:
                self._logger.warning(
                    f"{sum_small} very small or negative eigenvalues are detected on {self._dim}. "
                    f"To avoid numerical instability, these values are pruned. "
                    f"The new explicit dimension is now {self._dim - sum_small}.")
                keep_idx = torch.logical_not(idx_small)
                lambdas = lambdas[keep_idx]
                H = H[:, keep_idx]
                self._dim -= sum_small

            self._save(key="_nystrom_H", fun=lambda: H, level_key='_nystrom_elements')
            lambdas_sqrt = torch.sqrt(lambdas)
            self._save(key="_nystrom_lambdas_sqrt_inv",
                       fun=lambda: (torch.diag(1 / lambdas_sqrt)).data, level_key='_nystrom_elements')
            self._save(key="_nystrom_sample_phi",
                       fun=lambda: (H @ torch.diag(lambdas_sqrt)).data, level_key='_nystrom_elements')

    def update_sample(self, sample_values, idx_sample=None):
        raise NotImplementedError

    def _explicit_with_none(self, x=None):
        self._compute_decomposition()

        if x is None:
            return self._get(key="_nystrom_sample_phi")

        Kx = self._base_kernel.k(x)
        return Kx @ self._get(key="_nystrom_H") @ self._get(key="_nystrom_lambdas_sqrt_inv")

    def _explicit(self, x):
        # should never happen
        raise Exception("This should never happen, a bug must have occurred.")

    def _explicit_preimage(self, phi) -> torch.Tensor:
        raise utils.ExplicitError(cls=self, message='Explicit pre-image is not possible with the Nystrom kernel.')
