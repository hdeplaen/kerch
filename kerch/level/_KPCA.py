# coding=utf-8
import torch
from torch import Tensor as T
from abc import ABCMeta
from typing import Union, Iterator

from ._Level import _Level
from .. import utils


class _KPCA(_Level, metaclass=ABCMeta):
    r"""
    Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(_Level)
    def __init__(self, *args, **kwargs):
        kwargs['kernel_transform'] = kwargs.pop('kernel_transform', []).append('mean_centering')
        super(_KPCA, self).__init__(*args, **kwargs)
        self._vals = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=False)
        self._subloss_projected = None
        self._subloss_original = None

    @property
    def vals(self) -> T:
        r"""
        Eigenvalues of the model. The model has to be fitted for these values to exist.
        """
        return self._vals

    @vals.setter
    def vals(self, val):
        val = utils.castf(val, tensor=False, dev=self._vals.device)
        self._vals.data = val

    def total_variance(self, as_tensor=False, normalize=True, representation=None) -> Union[float, T]:
        r"""
        Total variance contained in the feature map. In primal formulation,
        this is given by :math:`\DeclareMathOperator{\tr}{tr}\tr(C)`, where :math:`C = \sum\phi(x)\phi(x)^\top` is
        the covariance matrix on the sample. In dual, this is given by :math:`\DeclareMathOperator{\tr}{tr}\tr(K)`,
        where :math:`K_{ij} = k(x_i,x_j)` is the kernel matrix on the sample.

        :param as_tensor: Indicated whether the variance has to be returned as a float or a torch.Tensor., defaults
            to ``False``
        :type as_tensor: bool, optional

        .. warning::
            For this value to strictly be interpreted as a variance, the corresponding kernel (or feature map)
            has to be normalized. In that case however, the total variance will amount to the dimension of the feature
            map in primal and the number of datapoints in dual.
        """
        representation = utils.check_representation(representation=representation, default=self._representation)
        level_key = "KPCA_total_variance_default_representation" if representation == self._representation \
            else "KPCA_total_variance_other_representation"
        if representation == 'primal':
            var = self._get("total_variance_primal", level_key=level_key, fun=lambda: torch.trace(self.C))
        else:
            var = self._get("total_variance_dual", level_key=level_key, fun=lambda: torch.trace(self.K))
        if normalize:
            var /= self.num_idx
        if as_tensor:
            return var
        return var.detach().cpu().numpy()

    def model_variance(self, as_tensor=False, normalize=True) -> Union[float, T]:
        r"""
        Total variance learnt by the model given by the sum of the eigenvalues.

        :param as_tensor: Indicated whether the variance has to be returned as a float or a torch.Tensor., defaults
            to ``False``
        :type as_tensor: bool, optional

        .. warning::
            For this value to strictly be interpreted as a variance, the corresponding kernel (or feature map)
            has to be normalized. We refer to the remark of ``total_variance``.
        """
        var = torch.sum(self.vals)
        if normalize:
            var /= self.num_idx
        if as_tensor:
            return var
        return var.detach().cpu().numpy()

    def _reset_hidden(self) -> None:
        super(_KPCA, self)._reset_hidden()
        self._remove_from_cache(["total_variance_primal", "total_variance_dual"])

    def relative_variance(self, as_tensor=False) -> Union[float, T]:
        r"""
        Relative variance learnt by the model given by ```model_variance``/``total_variance``.
        This number is always comprised between 0 and 1 and avoids any considerations on normalization.

        :param as_tensor: Indicated whether the variance has to be returned as a float or a torch.Tensor., defaults
            to ``False``
        :type as_tensor: bool, optional
        """
        var = self.model_variance(as_tensor=as_tensor, normalize=False) / \
              self.total_variance(as_tensor=as_tensor, normalize=False)
        return var

    ######################################################################################

    def _solve_primal(self) -> None:
        C = self.C

        if self._dim_output is None:
            self._dim_output = self.dim_feature
            self._log.warning(f"The output dimension has not been set and is now set to its maximum possible "
                              f"value. In primal formulation, this corresponds to the feature dimension "
                              f"(dim_output=dim_feature={self.dim_output}).")
        elif self._dim_output > self.dim_feature:
            self._log.warning(f"In primal, the output dimension {self.dim_output} (the number of "
                              f"eigenvectors) must not exceed the feature dimension {self.dim_feature} (the dimension "
                              f"of the correlation matrix to be diagonalized). As this is the case here, the output "
                              f"dimension is reduced to {self.dim_feature}.")
            self.dim_output = self.dim_feature

        v, w = utils.eigs(C, k=self.dim_output, psd=True)

        self.weight = w
        self.vals = v

    def _solve_dual(self) -> None:
        K = self.K

        if self._dim_output is None:
            self._dim_output = self.num_idx
            self._log.warning(f"The output dimension has not been set and is now set to its maximum possible "
                              f"value. In dual formulation, this corresponds to the number of datapoints in the "
                              f"current sample (dim_output=num_idx={self.dim_output}).")
        elif self._dim_output > self.num_idx:
            self._log.warning(f"In dual, the output dimension {self.dim_output} (the number of "
                              f"eigenvectors) must not exceed the number of samples {self.num_idx} (the dimension "
                              f"of the kernel matrix to be diagonalized). As this is the case here, the output "
                              f"dimension is reduced to {self.num_idx}.")
            self.dim_output = self.num_idx

        v, h = utils.eigs(K, k=self.dim_output, psd=True)

        self.update_hidden(h)
        self.vals = v

    @utils.extend_docstring(_Level.solve)
    @torch.no_grad()
    def solve(self, sample=None, target=None, representation=None, **kwargs) -> None:
        r"""
        Solves the model by decomposing the kernel matrix or the covariance matrix in principal components
        (eigendecomposition).
        """
        # KPCA models don't require the target to be defined. This is verified.
        if target is not None:
            self._log.warning("The target value is not used when fitting a KPCA model.")
        return _Level.solve(self,
                            sample=sample,
                            target=None,
                            representation=representation)

    def _stiefel_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        # the stiefel optimizer requires the first dimension to be the number of eigenvectors
        yield from super(_KPCA, self)._stiefel_parameters(recurse)
        if self._representation == 'primal':
            if self._weight_exists:
                yield self._weight
        else:
            if self._hidden_exists:
                yield self._hidden

    def loss(self, representation=None) -> T:
        r"""
        Reconstruction error on the sample.
        """
        return self._loss_original(representation=representation) \
            - self._loss_projected(representation=representation)

    def _loss_original(self, representation=None) -> T:
        representation = utils.check_representation(representation, self._representation, self)
        level_key = "Level_subloss_default_representation" if self._representation == representation \
            else "Level_subloss_representation"

        def fun():
            if representation == 'primal':
                M = self.C
            else:
                M = self.K
            return torch.trace(M)

        return self._get(key='subloss_original_' + representation,
                         level_key=level_key, fun=fun)

    def _loss_projected(self, representation=None) -> T:
        representation = utils.check_representation(representation, self._representation, self)
        level_key = "Level_subloss_default_representation" if self._representation == representation \
            else "Level_subloss_representation"

        def fun():
            if representation == 'primal':
                U = self._weight  # transposed compared to weight
                M = self.C
            else:
                U = self._hidden  # transposed compared to hidden
                M = self.K
            return torch.trace(U.T @ U @ M)

        return self._get(key='subloss_projected_' + representation,
                         level_key=level_key, fun=fun)

    def losses(self, representation=None) -> dict:
        return {'Original': self._loss_original().data.detach().cpu().item(),
                'Projected': self._loss_projected().data.detach().cpu().item(),
                **super(_KPCA, self).losses()}
