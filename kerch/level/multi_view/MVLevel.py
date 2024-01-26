import torch
from torch import Tensor as T
from abc import ABCMeta

from .._Level import _Level
from .MultiView import MultiView
from ... import utils


class MVLevel(_Level, MultiView, metaclass=ABCMeta):
    @utils.extend_docstring(MultiView)
    @utils.extend_docstring(_Level)
    def __init__(self, *args, **kwargs):
        super(MVLevel, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def solve(self, sample=None, target=None, representation=None, **kwargs) -> None:
        if sample is not None:
            self._logger.warning("It is not possible to directly change sample in multi-view models as for now. "
                              "The default values will be used.")

        return super(_Level, self).solve(sample=None,
                                         target=target,
                                         representation=representation,
                                         **kwargs)

    ####################################################################################################################

    def loss(self, representation=None) -> T:
        raise NotImplementedError