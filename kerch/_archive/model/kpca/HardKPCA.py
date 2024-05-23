"""
Hard KPCA Level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: April 2021
"""

import kerch._archive as mdl


class HardKPCA(mdl.kpca.KPCA):
    def __init__(self, **kwargs):
        super(HardKPCA, self).__init__(**kwargs)
        self.__eta = 0.

    def __str__(self):
        return f"Hard KPCA Level with {self._model['kernel'].__str__()} {super(HardKPCA, self).__str__()}."

    @property
    def hparams(self):
        return {"Constraint": "hard",
                **super(HardKPCA, self).hparams}

    def loss(self, x=None, y=None, idx_kernels=None):
        return super().loss(x, y)

    def hard(self, x, y):
        if x is None: x = self.layerin
        a, b = self.solve(x)
        self.linear.set(a, b)

    def projection(self):
        pass
