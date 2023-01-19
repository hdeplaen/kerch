"""
File containing the cosine kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

from .. import utils
from .linear import Linear


@utils.extend_docstring(Linear)
class Cosine(Linear):
    r"""
    Cosine kernel.

    .. math::
        k(x,y) = \frac{\cos\left(x^{\top} y\right)}{\max\left(\lVert x \rVert_2 \cdot \lVert y \rVert_2, \epsilon\right)}.

    This corresponds to a normalized linear kernel, or equivalently a linear kernel on which the datapoints are first
    projected onto a hypersphere.
    """

    def __init__(self, **kwargs):
        super(Cosine, self).__init__(**kwargs)

    @property
    def _required_projections(self):
        return "unit_sphere_normalization"

    def __str__(self):
        return "Cosine kernel."

    @property
    def hparams(self):
        return {"Kernel": "Cosine", **super(Cosine, self).hparams}