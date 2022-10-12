import torch
from torch import Tensor as T

from .level import Level
from .. import utils


class LSSVM(Level):
    r"""
    Least squares support vector machine.
    """

    @utils.extend_docstring(Level)
    @utils.kwargs_decorator({
        "gamma": 1.
    })
    def __init__(self, **kwargs):
        super(LSSVM, self).__init__(**kwargs)
        self._gamma = torch.nn.Parameter(torch.tensor(kwargs["gamma"], dtype=utils.FTYPE))
        self._mse_loss = torch.nn.MSELoss(reduction='mean')

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

    def _center_h(self):
        if self._hidden_exists:
            self._hidden.data -= torch.mean(self._hidden.data, dim=1)

    def _solve_primal(self) -> None:
        C = self.kernel.C
        phi = self.kernel.phi_sample
        dev = C.device
        dim_output = phi.shape[1]

        I = torch.eye(dim_output,
                      dtype=utils.FTYPE,
                      device=dev)
        N = torch.tensor([[self.num_sample]],
                         dtype=utils.FTYPE,
                         device=dev)

        P = torch.sum(phi, dim=0, keepdim=True)
        S = torch.sum(self.current_targets, dim=0, keepdim=True)
        Y = phi.t() @ self.current_targets

        A = torch.cat((torch.cat((C + (1 / self._gamma) * I, P.t()), dim=1),
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

        I = torch.eye(self.num_sample,
                      dtype=utils.FTYPE,
                      device=dev)

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
        N2 = self.current_targets

        A = torch.cat((torch.cat((K + (1 / self._gamma) * I, N1), dim=1),
                       torch.cat((N1.t(), Zero), dim=1)), dim=0)
        B = torch.cat((N2, Zeros), dim=0)

        sol = torch.linalg.solve(A, B)
        hidden = sol[0:-1].data
        bias = sol[-1].data

        self.hidden = hidden
        self.bias = bias

    def _euclidean_parameters(self, recurse=True):
        super(LSSVM, self)._euclidean_parameters(recurse)
        if self._representation == 'primal':
            if self._weight_exists:
                yield self._weight
        else:
            if self._hidden_exists:
                yield self._hidden

    def solve(self, sample=None, target=None, representation=None, **kwargs) -> None:
        # LS-SVMs require the target value to be defined. This is verified.
        return super(LSSVM, self).solve(sample=sample,
                                        target=target,
                                        representation=representation,
                                        **kwargs)

    def loss(self, representation=None) -> T:
        representation = utils.check_representation(representation, self._representation, self)
        pred = self.forward(representation=representation)
        mse_loss = self._mse_loss(pred, self.current_targets)
        if representation == 'primal':
            weight = self.weight
            reg_loss = torch.trace(weight.T @ weight)
            # replace by einsum to avoid unnecessary computations
            # torch.einsum('ij,ji',weight, weight)
        else:
            hidden = self.hidden
            reg_loss = torch.trace(hidden.T @ self.K @ hidden)
            # torch.einsum('ji,jk,ki',hidden,self.K,hidden)
        return reg_loss / self.num_idx + self.gamma * mse_loss

    def after_step(self) -> None:
        self._center_h()
