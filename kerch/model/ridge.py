from .lssvm import LSSVM
from .. import utils
import torch


class Ridge(LSSVM):
    @utils.extend_docstring(LSSVM)
    @utils.kwargs_decorator({
        "bias": False
    })
    def __init__(self, **kwargs):
        if kwargs["bias"]:
            kwargs["bias"] = False
            self._log.warning('A ridge regression has no bias term. '
                              'The bias parameter is overwritten to False.')
        super(Ridge, self).__init__(**kwargs)

    def _solve_primal(self, target=None) -> None:
        C = self.kernel.C
        phi = self.kernel.phi_sample
        dev = C.device
        dim_output = phi.shape[1]

        I = torch.eye(dim_output,
                      dtype=utils.FTYPE,
                      device=dev)
        Y = phi.t() @ target

        A = C + (1 / self._gamma) * I

        sol = torch.linalg.solve(A, Y)
        self.weight = sol

    def _solve_dual(self, target=None) -> None:
        K = self.kernel.K
        dev = K.device

        I = torch.eye(self.num_sample,
                      dtype=utils.FTYPE,
                      device=dev)

        A = K + (1 / self._gamma) * I

        sol = torch.linalg.solve(A, target)
        self.hidden = sol
