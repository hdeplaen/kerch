from typing import Iterator

import torch
from torch import Tensor as T

from .Level import Level
from ... import utils


class LSSVM(Level):
    r"""
    Least squares support vector machine.

    :param gamma: Regularization parameter of the LSSVM. Defaults to 1.
    :type gamma: float, optional
    """

    @utils.extend_docstring(Level)
    @utils.kwargs_decorator({
        "requires_bias": True
    })
    def __init__(self, *args, **kwargs):
        super(LSSVM, self).__init__(*args, **kwargs)
        gamma = kwargs.pop('gamma', 1.)
        self._gamma = torch.nn.Parameter(torch.tensor(gamma, dtype=utils.FTYPE))
        self._mse_loss = torch.nn.MSELoss()

    def __str__(self):
        return "LSSVM with " + Level.__str__(self)

    @property
    def gamma(self) -> float:
        return self._gamma.data.cpu().numpy().item()

    @gamma.setter
    def gamma(self, val):
        val = utils.castf(val, dev=self._gamma.device, tensor=False)
        self._gamma.data = val
        self._reset_hidden()
        self._reset_weight()

    def _center_hidden(self):
        if self._hidden_exists:
            self._hidden.data -= torch.mean(self._hidden.data, dim=1)
        else:
            self._log.debug("The hidden variables cannot be centered as they are not set.")

    def _solve_primal(self) -> None:
        C = self.kernel.C
        phi = self.kernel.phi_sample
        dev = C.device
        dim_output = phi.shape[1]

        N = torch.tensor([[self.num_sample]],
                         dtype=utils.FTYPE,
                         device=dev)

        P = torch.sum(phi, dim=0, keepdim=True)
        S = torch.sum(self.current_target, dim=0, keepdim=True)
        Y = phi.t() @ self.current_target

        A = torch.cat((torch.cat((C + self._gamma * self._I_primal, P.t()), dim=1),
                       torch.cat((P, N), dim=1)), dim=0)
        B = torch.cat((Y, S), dim=0)

        sol = torch.linalg.solve(A, B)
        weight = sol[0:-1].data
        bias = sol[-1].data

        self.weight = weight
        self.bias = bias

    def _solve_dual(self) -> None:
        K = self.kernel.K
        dev = K.device

        Ones = torch.ones((self.num_sample, 1),
                          dtype=utils.FTYPE,
                          device=dev)

        Zero = torch.zeros((1, 1),
                           dtype=utils.FTYPE,
                           device=dev)

        Zeros = torch.zeros((1, self.dim_output),
                            dtype=utils.FTYPE,
                            device=dev)

        N1 = Ones
        N2 = self.current_target

        A = torch.cat((torch.cat((K + self._gamma * self._I_dual, N1), dim=1),
                       torch.cat((N1.t(), Zero), dim=1)), dim=0)
        B = torch.cat((N2, Zeros), dim=0)

        sol = torch.linalg.solve(A, B)
        hidden = sol[0:-1].data
        bias = sol[-1].data

        self.update_hidden(hidden, idx_sample=self.idx)
        self.bias = bias

    def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield from super(LSSVM, self)._euclidean_parameters(recurse)
        if self._representation == 'primal':
            if self._weight_exists:
                yield self._weight
                yield self._bias
        else:
            if self._hidden_exists:
                yield self._hidden
                yield self._bias

    def loss(self, representation=None) -> T:
        fact = 1 / self.num_idx
        return fact * self._subloss_regularization(representation) \
            + self.gamma * self._subloss_mse(representation)

    def _subloss_regularization(self, representation=None):
        representation = utils.check_representation(representation, self._representation, self)
        level_key = "Level_subloss_default_representation" if self._representation == representation \
            else "Level_subloss_representation"

        def fun():
            if representation == 'primal':
                weight = self.weight
                return torch.einsum('ij,ji',weight, weight)
                # torch.trace(weight.T @ weight)
            else:
                hidden = self.hidden
                return torch.einsum('ji,jk,ki',hidden,self.K,hidden)
                # torch.trace(hidden.T @ self.K @ hidden)

        return self._cache.get(key='subloss_regularization_' + representation,
                               level_key=level_key, fun=fun)

    def _subloss_mse(self, representation=None):
        representation = utils.check_representation(representation, self._representation, self)
        level_key = "Level_subloss_default_representation" if self._representation == representation \
            else "Level_subloss_representation"

        def fun():
            pred = self._forward(representation=representation)
            return self._mse_loss(pred, self.current_target)
        return self._cache.get(key='subloss_mse_' + representation,
                               level_key=level_key, fun=fun)

    def sublosses(self, representation=None) -> dict:
        return {'Regularization': self._subloss_regularization().data.detach().cpu(),
                'MSE': self._subloss_mse().data.detach().cpu(),
                **super(LSSVM, self).sublosses()}


    def after_step(self) -> None:
        super(LSSVM, self).after_step()
        self._center_hidden()

    def _update_hidden_from_weight(self):
        raise NotImplementedError
