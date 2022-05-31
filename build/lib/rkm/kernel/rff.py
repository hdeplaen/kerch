"""
File containing the RFF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

from .. import utils
from .explicit import explicit, base


@utils.extend_docstring(base)
class rff(explicit):
    r"""
    Random Fourier Features kernel. (not implemented)
    """

    @utils.kwargs_decorator(
        {"network": None})
    def __init__(self, **kwargs):
        super(rff, self).__init__(**kwargs)
        raise NotImplementedError

    def __str__(self):
        return "rff kernel"

    def hparams(self):
        return {"Kernel": "Random Fourier Features", **super(rff, self).hparams}

    def _explicit(self, x=None):
        raise NotImplementedError
