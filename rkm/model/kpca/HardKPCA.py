"""
Hard KPCA level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: April 2021
"""

import rkm.model as mdl

class HardKPCA(mdl.kpca.KPCA):
    def __init__(self, **kwargs):
        super(HardKPCA, self).__init__(**kwargs)
        self.__eta = 0.

    def __str__(self):
        return f"Hard KPCA level with {self.__model['kernel'].__str__()}."

    def loss(self, x=None, y=None, idx_kernels=None):
        return super().loss(x, y)

    def before_step(self, x=None, y=None):
        assert x is not None, \
            "Tensors x must be specified before each step in a hard KPCA."
        a, b = self.solve(x)
        self.model["linear"].set(a, b)

    def after_step(self, x=None, y=None):
        pass