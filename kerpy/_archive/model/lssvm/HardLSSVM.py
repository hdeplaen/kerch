"""
Hard LS-SVM abstract level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from kerpy._archive.model.lssvm import LSSVM


class HardLSSVM(LSSVM):
    def __init__(self, **kwargs):
        super(HardLSSVM, self).__init__(**kwargs)
        # self._eta = 0.

    def __str__(self):
        return f"Hard LS-SVM level with {self.kernel.__str__()} {super(HardLSSVM, self).__str__()}."

    @property
    def hparams(self):
        return {"Constraint": "hard",
                **super(HardLSSVM, self).hparams}

    def loss(self, x=None, y=None):
        return super().loss(x, y)

    def hard(self, x=None, y=None):
        a, b = self.solve(x, y)
        self.linear.set(a, b)

    def projection(self):
        pass
