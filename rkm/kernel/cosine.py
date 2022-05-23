"""
File containing the cosine kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

from .. import utils
from .implicit import implicit, base

from torch.nn.functional import cosine_similarity
import torch



@utils.extend_docstring(base)
class cosine(implicit):
    r"""
    Cosine kernel.

    .. math::
        k(x,y) = \frac{\cos\left(x^{\top} y\right)}{\max\left(\lVert x \rVert_2 \cdot \lVert y \rVert_2, \epsilon\right)}.


    :param eps: Numerical stabilization :math:`\epsilon` to avoid the origin singularity., defaults to 1.e-8
    :type eps: double, optional
    """

    @utils.kwargs_decorator({
        "eps": 1.e-8
    })
    def __init__(self, **kwargs):
        super(cosine, self).__init__(**kwargs)
        self._eps = kwargs["eps"]

    def __str__(self):
        return "Cosine kernel."

    @property
    def eps(self):
        r"""
        Numerical stabilization :math:`\epsilon` to avoid the origin singularity.
        """
        return self._eps

    @eps.setter
    def eps(self, val):
        self._eps = val
        self._reset()

    @property
    def hparams(self):
        return {"Kernel": "Cosine", **super(cosine, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(cosine, self)._implicit(x_oos, x_sample)
        return cosine_similarity(x_oos, x_sample, dim=1, eps=self._eps)