"""
File containing the cosine kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

from .. import utils
from .linear import linear, base

import torch




@utils.extend_docstring(base)
class cosine(linear):
    r"""
    Cosine kernel.

    .. math::
        k(x,y) = \frac{\cos\left(x^{\top} y\right)}{\max\left(\lVert x \rVert_2 \cdot \lVert y \rVert_2, \epsilon\right)}.

    This corresponds to a normalized linear kernel, or equivalently a linear kernel on which the datapoints are first
    projected onto a hypersphere.
    """

    def __init__(self, **kwargs):
        super(cosine, self).__init__(**{**kwargs,
                                        "normalize":True})

    def __str__(self):
        return "Cosine kernel."

    @property
    def hparams(self):
        return {"Kernel": "Cosine", **super(cosine, self).hparams}

    @property
    def normalize(self) -> bool:
        r"""
        Indicates if the kernel has to be normalized. Changing this value leads to a recomputation of the statistics.
        """
        return self._normalize_requested

    @normalize.setter
    def normalize(self, val: bool):
        self._log.info('Changing the normalization has not effect on the cosine kernel as it is always normalized by '
                       'definition. Consider a linear kernel then.')