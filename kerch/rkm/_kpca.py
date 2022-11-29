import torch
from torch import Tensor as T
from abc import ABCMeta, abstractmethod
from typing import Union

from ._level import _Level
from kerch import utils


class _KPCA(_Level):
    r"""
    Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(_Level)
    @utils.kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(_KPCA, self).__init__(*args, **kwargs)
        self._vals = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=False)

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

    def total_variance(self, as_tensor=False, normalize=True) -> Union[float, T]:
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
        if "_total_variance" not in self._cache:
            if self._representation == 'primal':
                self._cache["_total_variance"] = torch.trace(self.C)
            else:
                self._cache["_total_variance"] = torch.trace(self.K)
        var = self._cache["_total_variance"]
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

    def relative_variance(self, as_tensor=False) -> Union[float, T]:
        r"""
        Relative variance learnt by the model given by ```model_variance``/``relative_variance``.
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

        if self.dim_output is None:
            self._dim_output = self.dim_feature
        elif self.dim_output > self.dim_feature:
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

        if self.dim_output is None:
            self._dim_output = self.num_idx
        elif self.dim_output > self.num_idx:
            self._log.warning(f"In dual, the output dimension {self.dim_output} (the number of "
                              f"eigenvectors) must not exceed the number of samples {self.num_idx} (the dimension "
                              f"of the kernel matrix to be diagonalized). As this is the case here, the output "
                              f"dimension is reduced to {self.num_idx}.")
            self.dim_output = self.num_idx

        v, h = utils.eigs(K, k=self.dim_output, psd=True)

        self.hidden = h
        self.vals = v

    @utils.extend_docstring(_Level.solve)
    def solve(self, sample=None, target=None, representation=None, **kwargs) -> None:
        r"""
        Solves the model by decomposing the kernel matrix or the covariance matrix in principal components
        (eigendecomposition).
        """
        # KPCA models don't require the target to be defined. This is verified.
        if target is not None:
            self._log.warning("The target value is discarded when fitting a KPCA model.")
        return _Level.solve(self,
                            sample=sample,
                            target=None,
                            representation=representation)

    ######################################################################################

    def _primal_obj(self, x=None) -> T:
        P = self.weight @ self.weight.T  # primal projector
        R = self._I_primal - P  # reconstruction
        C = self.c(x)  # covariance
        return torch.trace(R * C)  # reconstruction error on the covariance

    def _dual_obj(self, x=None) -> T:
        P = self.hidden @ self.hidden.T  # dual projector
        R = self._I_dual - P  # reconstruction
        K = self.k(x)  # kernel matrix
        return torch.trace(R * K)  # reconstruction error on the kernel

    ######################################################################################

    def _stiefel_parameters(self, recurse=True):
        # the stiefel optimizer requires the first dimension to be the number of eigenvectors
        super(_KPCA, self)._stiefel_parameters(recurse)
        if self._representation == 'primal':
            if self._weight_exists:
                yield self._weight
        else:
            if self._hidden_exists:
                yield self._hidden

    def optimize(self, **kwargs) -> None:
        super(_KPCA, self).optimize(**kwargs)

        # once the optimization is done, the eigenvalues still have to be defined
        representation = utils.check_representation(kwargs["representation"], self._representation, cls=self)
        if representation == 'primal':
            self.vals = torch.diag(self.weight.T @ self.C @ self.weight)
        else:
            self.vals = torch.diag(self.hidden.T @ self.K @ self.hidden)

    @utils.kwargs_decorator({
        "representation": None
    })
    def fit(self, **kwargs):
        if not self.attached:
            if self.dim_output is None:
                representation = utils.check_representation(kwargs["representation"], self._representation, cls=self)
                if representation == "primal":
                    self._dim_output = self.dim_feature
                else:
                    self._dim_output = self.num_idx
            super(_KPCA, self).fit(**kwargs)

    ####################################################################################################################

    def loss(self, representation=None) -> T:
        r"""
        Reconstruction error on the sample.
        """
        representation = utils.check_representation(representation, self._representation, self)
        if representation == 'primal':
            U = self._weight    # transposed compared to weight
            M = self.C
        else:
            U = self._hidden    # transposed compared to hidden
            M = self.K
        loss = torch.trace(M) - torch.trace(U.T @ U @ M)
        return loss


