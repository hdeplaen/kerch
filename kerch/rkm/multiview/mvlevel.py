from torch import Tensor as T
from abc import ABCMeta

from kerch.rkm._level import _Level
from .multiview import MultiView
from kerch import utils


class MVLevel(_Level, MultiView, metaclass=ABCMeta):
    @utils.extend_docstring(MultiView)
    @utils.extend_docstring(_Level)
    def __init__(self, *args, **kwargs):
        super(MVLevel, self).__init__(*args, **kwargs)

    def solve(self, sample=None, target=None, representation=None) -> None:
        if sample is not None:
            self._log.warning("It is not possible to directly change sample in multi-view models as for now. "
                              "The default values will be used.")

        return super(_Level, self).solve(sample=None,
                                         target=target,
                                         representation=representation)

    ####################################################################################################################

    def loss(self, representation=None) -> T:
        raise NotImplementedError