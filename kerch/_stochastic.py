"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from abc import ABCMeta, abstractmethod
from torch import Tensor

from . import utils
from ._cache import _Cache
from ._logger import _Logger


class _Stochastic(_Cache,  # creates a transportable cache (e.g. for GPU)
                  metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super(_Stochastic, self).__init__(*args, **kwargs)
        self._num_total = None
        self._idx_stochastic = None

    @property
    def num_idx(self) -> int:
        r"""
        Number of selected indices when performing various operations. This is only relevant in the case of stochastic
        training.
        """
        return len(self._idx_stochastic)

    @property
    def idx(self) -> Tensor:
        r"""
        Indices used when performing various operations. This is only relevant in the case of stochastic training.
        """
        return self._idx_stochastic

    @property
    def _all_idx(self):
        return range(self._num_total)

    def train(self, mode=True):
        r"""
        Activates the training mode, which disables the gradients computation and disables stochasticity. For the
        gradients and other things, we refer to the `torch.nn.Module` documentation. For the stochastic part, when put
        in evaluation mode (`False`), all the sample points are used for the computations, regardless of
        the previously specified indices.
        """
        if not mode:
            self.stochastic()
        return self

    def stochastic(self, idx=None, prop=None):
        """
        Resets which subset of the samples are to be used until the next call of this function. This is relevant in the
        case of stochastic training.

        :param idx: Indices of the sample subset relative to the original sample set., defaults to `None`
        :type idx: int[], optional
        :param prop: Instead of giving indices, passing a proportion of the original sample set is also
            possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
            such that :math:`0 <` `prop_stochastic` :math:`\leq 1`., defaults to `None`.
        :type prop: double, optional

        If `None` is specified for both `idx_stochastic` and `prop_stochastic`, all samples are used and the subset equals the
        original sample set. This is also the default behavior if this function is never called, nor the parameters
        specified during initialization.

        .. note::
            Both `idx_stochastic` and `prop_stochastic` cannot be filled together as conflict would arise.
        """
        if self._num_total is None:
            self._log.info("Setting stochastic indices cannot work before any data or dimensions have been fed.")
            return

        self._reset()
        assert idx is None or prop is None, "Both idx_stochastic and prop_stochastic are not None. " \
                                                          "Please choose one non-None parameter only."

        if idx is not None:
            self._log.debug("Using the provided indices for stochasticity.")
            self._idx_stochastic = idx
        elif prop is not None:
            assert prop <= 1., 'Parameter prop_stochastic: the chosen proportion cannot be greater than 1.'
            assert prop > 0., 'Parameter prop_stochastic: the chosen proportion must be strictly greater than 0.'

            self._log.debug("Randomly defining the indices.")
            n = self._num_total
            k = round(n * prop)
            perm = torch.randperm(n)
            self._idx_stochastic = perm[:k]
        else:
            self._log.debug("Using all indices.")
            self._idx_stochastic = self._all_idx

        for stochastic_module in self.children():
            if isinstance(stochastic_module, _Stochastic):
                stochastic_module.stochastic(idx=self._idx_stochastic)
