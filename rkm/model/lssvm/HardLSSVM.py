"""
Hard LS-SVM abstract level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.model.lssvm as lssvm

class HardLSSVM(lssvm.LSSVM):
    def __init__(self, **kwargs):
        super(HardLSSVM, self).__init__(**kwargs)
        self.__eta = 0.

    def __str__(self):
        return f"Hard LS-SVM level with {self.__model['kernel'].__str__()}."

    def loss(self, x=None, y=None, idx_kernels=None):
        return super().loss(x, y)

    def before_step(self, x=None, y=None):
        assert (x is not None) and (y is not None), \
            "Tensors x and y must be specified before each step in a hard LSSVM."
        a, b = self.solve(x, y)
        self.model["linear"].set(a, b)

    def after_step(self, x=None, y=None):
        pass