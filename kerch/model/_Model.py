from __future__ import annotations
from abc import ABCMeta, abstractmethod

from typing import Iterator
import torch

from ..level import Level, factory
from ..utils import NotInitializedError, kwargs_decorator
from .._module._Stochastic import _Stochastic


class _Model(_Stochastic, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(_Model, self).__init__(*args, **kwargs)
        self._levels: list[Level] = list()

    @property
    def levels(self) -> Iterator[Level]:
        yield from self._levels

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    def _check_levels(self):
        if self.num_levels == 0:
            raise NotInitializedError(cls=self, message="The model does not contain any level.")

    def level(self, num: int) -> Level:
        assert num < self.num_levels, f"Model has less levels ({self.num_levels}) than requested ({int})."
        return next(x for i, x in enumerate(self.levels) if i == num)

    @property
    def _first_level(self) -> Level:
        self._check_levels()
        return next(self.levels)

    @property
    def _last_level(self) -> Level:
        self._check_levels()
        return self._levels[-1]

    @property
    def dim_input(self) -> int:
        return self._first_level.dim_input

    @property
    def dim_output(self) -> int:
        return self._last_level.dim_output

    @property
    def num_sample(self) -> int:
        return self._first_level.num_sample

    def init_sample(self, sample=None):
        self._first_level.init_sample(sample=sample)
        self._num_total = self._first_level.num_sample
        self.stochastic(idx=self._first_level.idx)

    @property
    def empty_sample(self) -> bool:
        return self._first_level.empty_sample

    def init_levels(self, full: bool = False) -> None:
        r"""
        Initializes all levels based on the model sample.

        :param full: If specified to False, the sample initialization is going to randomly initialized for the deeper
            levels to avoid running a forward on the full data. This is relevant in a stochastic setting. Defaults
            to True.
        :type full: bool, optional
        """
        if full:
            x = self._first_level()
            for level in self.levels[1:]:
                level.init_sample(sample=x)
                level.init_parameters(overwrite=False)
                x = level()
        else:
            for level in self.levels[:1]:
                level.num_sample = self.num_sample
                level.init_parameters(overwrite=False)

    def init_target(self, target: torch.Tensor, num_level: int = -1) -> None:
        if num_level == -1:
            level = self._last_level
        else:
            level = self.level(num_level)
        try:
            level.target = target
        except NotInitializedError:
            raise NotInitializedError(cls=self, message="Please initialize the sample and the levels first.")

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if self.training:
            x = self.current_sample_projected
            for level in self.levels:
                level.update_sample(x, idx_sample=self._idx_stochastic)
                if level.param_trainable is False:
                    level.solve()
                x = level()
        else:
            for level in self.levels:
                x = level(x)
        return x

    def loss(self):
        return sum([level.eta * level.loss() for level in self.levels])

    def stochastic(self, idx=None, prop=None):
        super().stochastic(idx, prop)
        for level in self.levels:
            level.stochastic(idx=self._idx_stochastic)

    def after_step(self):
        for level in self.levels:
            level.after_step()

    def _stiefel_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        for level in self.levels:
            yield from level._stiefel_parameters(recurse=recurse)

    def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        for level in self.levels:
            yield from level._euclidean_parameters(recurse=recurse)

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        for level in self.levels:
            yield from level._slow_parameters(recurse=recurse)

    def train(self, mode=True):
        super().train(mode)
        for level in self.levels:
            level.train(mode)

    @kwargs_decorator({"level_type": None,
                       "constraint": "soft"})
    def _append_level(self, *args, **kwargs) -> None:
        # create level and remove sample
        level = factory(*args, **kwargs)

        # verify the correct assignment of the output dimension
        try:
            _ = level.dim_output
        except NotInitializedError:
            raise AssertionError("The argument dim_output is not specified. This is not required target are "
                                 "explicitly specified during initialization.")

        # verify the correct assignment of the input dimension
        try:
            _ = level.dim_input
        except NotInitializedError:
            if self.num_levels > 0:
                kwargs["dim_input"] = self.level(self.num_levels - 1).dim_output


        # overwrite param_trainable
        match kwargs['constraint']:
            case "soft":
                level.param_trainable = True
            case "hard":
                level.param_trainable = False
            case _:
                raise NameError('The constraint argument must be either soft or hard.')



        # verify dimension consistency
        if self.num_levels > 0:
            assert self.level(self.num_levels - 1).dim_output == level.dim_input, \
                f"The input dimension of the level to be added ({level.dim_input}) does not correspond to the " \
                f"output dimension of the previous level ({self.level(self.num_levels - 1).dim_output})."

        self._levels.append(level)

    def reset(self, children=False) -> None:
        for level in self.levels:
            level.reset(children=children)
