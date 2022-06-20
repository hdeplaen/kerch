"""
Soft KPCA Level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: April 2021
"""

import kerch._archive as mdl

class SoftKPCA(mdl.kpca.KPCA):
    def __init__(self, **kwargs):
        super(SoftKPCA, self).__init__(**kwargs)

    def __str__(self):
        return f"Soft KPCA Level with {self._model['kernel'].__str__()} {super(SoftKPCA, self).__str__()}."

    @property
    def hparams(self):
        return {"Constraint": "soft",
                **super(SoftKPCA, self).hparams}

    def evaluate(self, x):
        return super(SoftKPCA, self).evaluate(x)

    def loss(self, x=None, y=None):
        return super(SoftKPCA, self).loss(x, y)

    def hard(self, x=None, y=None):
        pass

    def projection(self, x=None, y=None):
        pass