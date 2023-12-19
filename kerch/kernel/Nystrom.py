"""
File containing the feature kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch


from .. import utils
from ._Explicit import _Explicit, _Projected
from ._Base import _Base
from .factory import factory

@utils.extend_docstring(_Projected)
class Nystrom(_Explicit):
    r"""
    Nyström kernel. Constructs an explicit feature map based on the eigendecomposition of any kernel matrix based on
    some sample.

    :param dim: Dimension of the explicit feature map to be constructed. This value cannot exceed the number of sample
        points. During eigendecomposition, very small eigenvalues are also going to pruned to avoid numerical
        instability. If `None`, the value will be assigned to `num_sample`., defaults to `None`
    :param base_type: The name of kernel on which the explicit feature map is going to be constructed., defaults to
        `"rbf"`
    :param base_center: Specifies if the _Projected kernel has to be centered. This is redundant and can be directly handled
        by the Nystrom kernel itself. It is only added for completeness., defaults to `False`
    :param base_normalize: Specifies if the _Projected kernel has to be normalized., This is redundant and can be directly
        handled by the Nystrom kernel itself. It is only added for completeness., defaults to `False`
    :param \**kwargs: Other arguments for the _Projected kernel (e.g. the bandwidth for an RBF kernel, the degree for a
        polynomial kernel etc.). For the default values, please refer to the requested class in question.
    :param base_kernel: Instead of creating a new kernel on which to use the Nyström method, one can also perform it
        on an existing kernel. In that case, the other _Projected arguments are bypassed., defaults to `None`
    :type dim: int, optional
    :type \**kwargs: dict, optional
    :type base_type: str, optional
    :type base_center: bool, optional
    :type base_normalize: bool, optional
    :type base_kernel: kerch.kernel.*, optional
    """

    @utils.kwargs_decorator({
        "dim": None,
        "base_type": "rbf",
        "base_center": False,
        "base_normalize": False,
        "base_kernel": None,
        "base_kernel_projections": []
    })
    def __init__(self, **kwargs):
        assert kwargs["base_type"].lower() != "nystrom", 'Cannot create a Nyström kernel based on another Nyström ' \
                                                         'kernel.'
        self._base_kernel = None

        k = kwargs["base_kernel"]
        assert not isinstance(k, str), "base_kernel must be of kernel type (use base_type instead)."
        if k is None:
            # normal case with a kernel created from the factory
            super(Nystrom, self).__init__(**kwargs)

            self._base_kernel = factory(**{**kwargs,
                                               "_center": kwargs["base_center"],
                                               "_normalize": kwargs["base_normalize"],
                                               "name": kwargs["base_type"],
                                               "kernel_projections": kwargs["base_kernel_projections"]})
            self._base_kernel.init_sample(sample=self.current_sample, idx_sample=self.idx)
        else:
            # nystromizing some existing kernel
            assert isinstance(k, _Base), "The provided kernel is not of the kernel class."
            super(Nystrom, self).__init__(**{**kwargs,
                                             "sample":k.sample,
                                             "sample_trainable": k.sample_trainable,
                                             "idx_sample": k.idx})
            self._base_kernel = k
            self._log.info("Keeping original kernel projection (no overwriting, so base_kernel_projections is "
                           "neglected).")

        self._dim = kwargs["dim"]
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
        self._reset_cache()

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
        assert self._base_kernel is not None, 'Base kernel has not been defined yet.'
        return self._base_kernel

    def hparams(self):
        return {"Kernel": "Feature", **super(Nystrom, self).hparams}

    def init_sample(self, sample=None, idx_sample=None, prop_sample=None):
        super(Nystrom, self).init_sample(sample=sample, idx_sample=idx_sample, prop_sample=prop_sample)
        if self._base_kernel is not None:
            self._base_kernel.init_sample(sample=self.current_sample, idx_sample=self.idx)

    def _compute_decomposition(self):
        if "H" not in self._cache:
            self._log.info("Computing eigendecomposition for Nystrom kernel.")

            if self._dim is None:
                self.dim = self._num_total

            K = self._base_kernel.K
            lambdas, H = utils.eigs(K, k=self._dim)

            # verify that the decomposed kernel is PSD
            sum_neg = torch.sum(lambdas < 0)
            if sum_neg > 0:
                self._log.warning(f"The decomposed kernel is not positive semi-definite as it possesses {sum_neg} "
                                  f"negative eigenvalues. These will be discarded, but may prove relevant if their "
                                  f"magnitude is non-negligible.")

            # prune very small eigenvalues if they exist to avoid unstability due to the later inversion
            idx_small = lambdas < 1.e-10
            sum_small = torch.sum(idx_small)
            if sum_small > 0:
                self._log.warning(
                    f"{sum_small} very small or negative eigenvalues are detected on {self._dim}. "
                    f"To avoid numerical instability, these values are pruned. "
                    f"The new explicit dimension is now {self._dim - sum_small}.")
                keep_idx = torch.logical_not(idx_small)
                lambdas = lambdas[keep_idx]
                H = H[:, keep_idx]
                self._dim -= sum_small

            self._cache["lambdas"] = lambdas
            self._cache["H"] = H
            self._cache["lambdas_sqrt"] = torch.sqrt(lambdas)
            self._cache["sample_phi"] = (H @ torch.diag(self._cache["lambdas_sqrt"])).data

    def update_sample(self, sample_values, idx_sample=None):
        raise NotImplementedError

    def _explicit_with_none(self, x=None):
        self._compute_decomposition()

        if x is None:
            return self._cache["sample_phi"]

        Ky = self._base_kernel.k(x)
        return Ky @ self._cache["H"] @ torch.diag(1 / self._cache["lambdas_sqrt"])

    def _explicit(self, x):
        pass # should never happen

    def _explicit_preimage(self, phi) -> torch.Tensor:
        raise NotImplementedError