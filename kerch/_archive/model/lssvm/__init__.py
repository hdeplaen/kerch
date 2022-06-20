"""
LS-SVM abstract Level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from abc import ABCMeta

import kerch
from kerch._archive.model.level import Level
from kerch._archive import RepresentationError


class LSSVM(Level, metaclass=ABCMeta):
    """
    Abstract LSSVM class.
    """

    @kerch.kwargs_decorator(
        {"gamma": 1.,
         "centering": False,
         "requires_bias": True})
    def __init__(self, **kwargs):
        """

        :param gamma: reconstruction / regularization trade-off.
        """
        super(LSSVM, self).__init__(**kwargs)

        self._gamma = kwargs["gamma"]
        self._criterion = torch.nn.MSELoss(reduction="mean")
        self._generate_representation(**kwargs)

        self._last_recon = torch.tensor(0.)
        self._last_reg = torch.tensor(0.)
        self._last_loss = torch.tensor(0.)

    @property
    def gamma(self):
        return self._gamma

    @property
    def hparams(self):
        return {"Type": "LSSVM",
                "Gamma": self.gamma,
                "Classifier": self._classifier,
                **super(LSSVM, self).hparams}

    @property
    def last_recon(self):
        return self._last_recon.data

    @property
    def last_reg(self):
        return self._last_reg.data

    @property
    def last_loss(self):
        return self._last_loss.data

    def Omega(self):
        K = self.kernel.dmatrix()
        if self._classifier:
            K = self._y @ self._y.t() * K
        return K

    def _generate_representation(self, **kwargs):
        # REGULARIZATION
        def primal_reg():
            idx_kernels = self._idxk.idx_kernels
            weight = self.linear.weight
            return (1 / len(idx_kernels)) * weight.t() @ weight
            # return weight.t() @ weight

        def dual_reg():
            idx_kernels = self._idxk.idx_kernels
            alpha = self.linear.alpha[idx_kernels]
            K = self.kernel.dmatrix()
            return (1 / len(idx_kernels)) * alpha.t() @ self.Omega() @ alpha
            # return alpha.t() @ K @ alpha

        switcher_reg = {"primal": primal_reg,
                        "dual": dual_reg}
        self._reg = switcher_reg.get(kwargs["representation"], RepresentationError)

    def evaluate(self, x, all_kernels=False):
        x = super(LSSVM, self).evaluate(x, all_kernels=all_kernels)
        if self._classifier: x = torch.sign(x)
        return x

    def recon(self, x, y):
        x_tilde = self.forward(x, y)
        try:
            if self._classifier:
                r = self._criterion(y * x_tilde, torch.abs(y))
            else:
                r = self._criterion(x_tilde, y)
        except:
            raise Exception('Probably input and output which are not of the same size.')
        return r, x_tilde

    def reg(self):
        idx_kernels = self._idxk.idx_kernels
        return torch.trace(self._reg())

    def loss(self, x=None, y=None):
        # print(self.linear.weight.t())

        recon, x_tilde = self.recon(x, y)
        reg = self.reg()

        l = .5 * reg + .5 * self._gamma * recon
        # print(f"REG:{reg} | RECON:{recon} | LOSS:{l}")

        self._last_recon = recon.data
        self._last_reg = reg.data
        self._last_loss = l.data
        return l, x_tilde

    def solve(self, x, y=None):
        assert y is not None, "Tensor y is unspecified. This is not allowed for a LSSVM Level."
        switcher = {'primal': lambda: self.primal(x, y),
                    'dual': lambda: self.dual(x, y)}

        return switcher.get(self.representation, RepresentationError)()

    def primal(self, x, y):
        C, phi = self.kernel.pmatrix()
        N, n = phi.shape
        I = torch.eye(n, device=self.device)
        N = torch.tensor([[N]], device=self.device)
        P = torch.sum(phi, dim=0, keepdim=True)
        S = torch.sum(y, dim=0, keepdim=True)
        Y = phi.t() @ y

        A = torch.cat((torch.cat((C + (1 / self._gamma) * I, P.t()), dim=1),
                       torch.cat((P, N), dim=1)), dim=0)
        B = torch.cat((Y, S), dim=0)

        sol, _ = torch.solve(B, A)
        # sol = torch.linalg.solve(A, B) # torch >= 1.8
        weight = sol[0:-1]
        bias = sol[-1].data

        reg = (1 / N) * weight.t() @ weight
        yhat = phi @ weight + bias
        recon = (1 / N) * torch.sum((yhat - y) ** 2)

        return weight, bias

    def dual(self, x, y):
        assert y.shape[1] == 1, "Not implemented for multi-dimensional output (as for now)."

        n = x.size(0)
        K = self.kernel.dmatrix()
        I = torch.eye(n, device=self.device)

        if self._classifier:
            N1 = y
            N2 = torch.ones((n, 1), device=self.device)
        else:
            N1 = torch.ones((n, 1), device=self.device)
            N2 = y

        A = torch.cat((torch.cat((self.Omega() + (1 / self._gamma) * I, N1), dim=1),
                       torch.cat((N1.t(), torch.tensor([[0.]], device=self.device)), dim=1)), dim=0)
        B = torch.cat((N2, torch.tensor([[0.]], device=self.device)), dim=0)

        sol, _ = torch.solve(B, A)
        # sol = torch.linalg.solve(A, B[:, None]) # torch >= 1.8
        alpha = sol[0:-1].data
        beta = sol[-1].data

        reg = (1 / len(y)) * alpha.t() @ K @ alpha
        if self._classifier:
            yhat = (K @ (y * alpha) + beta.expand([n, 1]))
        else:
            yhat = (K @ alpha + beta.expand([n, 1]))
        recon = (1 / len(y)) * torch.sum((yhat - y) ** 2)

        l = .5 * reg + .5 * self._gamma * recon
        # print(f"REG:{reg} | RECON:{recon} | LOSS:{l}")

        return alpha, beta

    def get_params(self, slow_names=None):
        euclidean = torch.nn.ParameterList(
            [p for n, p in self.kernel.named_parameters() if p.requires_grad and n not in slow_names])
        euclidean.extend(
            [p for p in self.linear.parameters() if p.requires_grad])
        slow = torch.nn.ParameterList(
            [p for n, p in self.kernel.named_parameters() if p.requires_grad and n in slow_names])
        stiefel = torch.nn.ParameterList()
        return euclidean, slow, stiefel
