"""
KPCA Level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import kerch
from kerch._archive.model.level import Level
from kerch._archive import RepresentationError
from kerch.utils import eigs
import torch
from abc import ABCMeta


class KPCA(Level, metaclass=ABCMeta):
    @kerch.kwargs_decorator(
        {"centering": True,
         "requires_bias": False})
    def __init__(self, **kwargs):
        """

        :param centering: True if input and kernel are centered (False by default).
        """
        super(KPCA, self).__init__(**kwargs)
        assert self._classifier is False, "No formulation exists for a KPCA classifier Level."

        self._generate_representation(**kwargs)

        self._last_var = torch.tensor(0.)

    @property
    def hparams(self):
        return {"Type": "KPCA",
                "Centering": self._centering,
                **super(KPCA, self).hparams}

    @property
    def last_loss(self):
        return self._last_var.data

    def _generate_representation(self, **kwargs):
        # REGULARIZATION
        def primal_var():
            C, _ = self.kernel.pmatrix()
            V = self.linear.weight
            # return torch.trace(C) - torch.trace(V.t() @ C @ V)
            return torch.trace(C) - torch.trace(V @ V.t() @ C)

        def dual_var():
            K = self.kernel.dmatrix()
            idx_kernels = self._idxk._idx_sample
            H = self.linear.alpha[idx_kernels, :]
            # l = torch.trace(K) - torch.trace(_H @ _H.t() @ K)
            # if l<0:
            #     print(f"{self.kernel.sigma} | {l}")
            # return l
            return torch.trace(K) - torch.trace(H @ H.t() @ K)
            # return (torch.trace(K) - torch.trace(_H @ _H.t() @ K)) / torch.abs(torch.sum(K, (0, 1)))

        switcher_var = {"primal": primal_var,
                        "dual": dual_var}
        self._var = switcher_var.get(kwargs["representation"], RepresentationError)

    def loss(self, x=None, y=None):
        x = self.forward(x, y)
        var = self._var()
        self._last_var = var.data
        return var, x

    def solve(self, x, y=None):
        switcher = {'primal': lambda: self.primal(x),
                    'dual': lambda: self.dual(x)}

        return switcher.get(self.representation, RepresentationError)()

    def primal(self, x, y=None):
        C, _ = self.kernel.pmatrix()
        s, v = eigs(C, k=self._size_out)
        w = v

        return w.data, None

    def dual(self, x, y=None):
        K = self.kernel.dmatrix()
        s, v = eigs(K, k=self._size_out)

        return v.data, None

    def get_params(self, slow_names=None):
        euclidean = torch.nn.ParameterList(
            [p for n, p in self.kernel.named_parameters() if p.requires_grad and n not in slow_names])
        slow = torch.nn.ParameterList(
            [p for n, p in self.kernel.named_parameters() if p.requires_grad and n in slow_names])
        stiefel = self._model['linear'].parameters() #bias term is also going to appear, but is equal to zero and not optimized
        return euclidean, slow, stiefel
