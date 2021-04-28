"""
Soft KPCA level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: April 2021
"""

import rkm.model as mdl

class SoftKPCA(mdl.kpca.KPCA):
    def __init__(self, **kwargs):
        super(SoftKPCA, self).__init__(**kwargs)

    def __str__(self):
        return f"Soft KPCA level with {self.__model['kernel'].__str__()}."

    def forward(self, x, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.__stoch_kernels
        return super().forward(x, idx_kernels=idx_kernels)

    def loss(self, x=None, y=None, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.__stoch_kernels
        return super().loss(x, y, idx_kernels=idx_kernels)

    def before_step(self, x=None, y=None):
        pass

    def after_step(self, x=None, y=None):
        pass