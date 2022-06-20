"""
Soft LS-SVM abstract Level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import kerch
import kerch._archive.model.lssvm as lssvm

class SoftLSSVM(lssvm.LSSVM):
    @kerch.kwargs_decorator(
        {"stochastic": 1.})
    def __init__(self, **kwargs):
        super(SoftLSSVM, self).__init__(**kwargs)
        self._stochastic = kwargs["stochastic"]

    def __str__(self):
        return f"Soft LS-SVM Level with {self._model['kernel'].__str__()} {super(SoftLSSVM, self).__str__()}."

    @property
    def hparams(self):
        return {"Constraint": "soft",
                **super(SoftLSSVM, self).hparams}

    def evaluate(self, x):
        return super(SoftLSSVM, self).evaluate(x)

    def loss(self, x=None, y=None):
        return super().loss(x, y)

    def recon(self, x, y):
        return super().recon(x, y)

    def reg(self):
        return super().reg()

    def hard(self, x, y):
        pass

    def projection(self):
        self.linear.project()
