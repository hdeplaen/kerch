import torch
from torch import Tensor as T

from .level import level
from .. import utils


class lssvm(level):
    r"""
    Least squares support vector machine.
    """

    @utils.extend_docstring(level)
    @utils.kwargs_decorator({
        "gamma": 1.
    })
    def __init__(self, **kwargs):
        super(lssvm, self).__init__(**kwargs)
        self._gamma = torch.nn.Parameter(torch.tensor(kwargs["gamma"], dtype=utils.FTYPE))

    @property
    def gamma(self) -> float:
        return self._gamma.data.cpu().numpy().item()

    @gamma.setter
    def gamma(self, val):
        val = utils.castf(val, dev=self._gamma.device, tensor=False)
        self._gamma.data = val
        self._reset_hidden()
        self._reset_weight()

    def _solve_primal(self, target=None) -> None:
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
        S = torch.sum(target, dim=0, keepdim=True)
        Y = phi.t() @ target

        A = torch.cat((torch.cat((C + (1 / self._gamma) * I, P.t()), dim=1),
                       torch.cat((P, N), dim=1)), dim=0)
        B = torch.cat((Y, S), dim=0)

        sol = torch.linalg.solve(A, B)
        weight = sol[0:-1].data
        bias = sol[-1].data

        self.weight = weight
        self.bias = bias

    def _solve_dual(self, target=None) -> None:
        K = self.kernel.K
        dev = K.device
        dim_output = target.shape[1]

        I = torch.eye(self.num_sample,
                      dtype=utils.FTYPE,
                      device=dev)

        Ones = torch.ones((self.num_sample, 1),
                          dtype=utils.FTYPE,
                          device=dev)

        Zero = torch.zeros((1,1),
                            dtype=utils.FTYPE,
                            device=dev)

        Zeros = torch.zeros((1,dim_output),
                            dtype=utils.FTYPE,
                            device=dev)

        N1 = Ones
        N2 = target

        A = torch.cat((torch.cat((K + (1 / self._gamma) * I, N1), dim=1),
                       torch.cat((N1.t(), Zero), dim=1)), dim=0)
        B = torch.cat((N2, Zeros), dim=0)

        sol = torch.linalg.solve(A, B)
        hidden = sol[0:-1].data
        bias = sol[-1].data

        self.hidden = hidden
        self.bias = bias


    def solve(self, sample=None, target=None, representation=None) -> None:
        # LS-SVMs require the target value to be defined. This is verified.
        if target is None:
            self._log.error("The target value should be provided when fitting an LS-SVM.")
            raise ValueError
        return super(lssvm, self).solve(sample=sample,
                                        target=target,
                                        representation=representation)