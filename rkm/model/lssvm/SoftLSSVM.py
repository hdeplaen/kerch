"""
Soft LS-SVM abstract level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model.lssvm as lssvm

class SoftLSSVM(lssvm.LSSVM):
    @rkm.kwargs_decorator(
        {"stochastic": 1.})
    def __init__(self, **kwargs):
        super(SoftLSSVM, self).__init__(**kwargs)
        self._stochastic = kwargs["stochastic"]

    def __str__(self):
        return f"Soft LS-SVM level with {self._model['kernel'].__str__()} {super(SoftLSSVM, self).__str__()}."

    @property
    def hparams(self):
        return {"Constraint": "soft",
                **super(SoftLSSVM, self).hparams}

    def forward(self, x, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self._stoch_kernels()
        return super().forward(x, idx_kernels=idx_kernels)

    def evaluate(self, x):
        return super(SoftLSSVM, self).evaluate(x)

    def loss(self, x=None, y=None, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self._stoch_kernels()
        return super().loss(x, y, idx_kernels=idx_kernels)

    def recon(self, x, y, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self._stoch_kernels()
        return super().recon(x, y, idx_kernels=idx_kernels)

    def reg(self, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self._stoch_kernels()
        return super().reg(idx_kernels=idx_kernels)

    def before_step(self, x=None, y=None):
        pass

    def after_step(self, x=None, y=None):
        self._model["linear"].project()
