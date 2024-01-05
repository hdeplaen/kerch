from typing import Iterator
import torch

from ..level import Level, factory
from ..utils import kwargs_decorator, NotInitializedError
from .._Sample import _Sample


class RKM(_Sample):
    def __init__(self, *args, **kwargs):
        super(RKM, self).__init__(*args, **kwargs)
        self._levels: list[Level] = list()

    def __repr__(self):
        description = f"RKM with following levels:"
        nl, tab = '\n', '\t'
        for num, level in enumerate(self.levels):
            dim_input = level.dim_input
            dim_output = level.dim_output
            description += f"{nl}{tab}* L{num}[{dim_input},{dim_output}]: {level.representation.capitalize()} {level}"
        return description

    @property
    def levels(self) -> Iterator[Level]:
        for level in self._levels:
            yield level

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    def level(self, num: int) -> Level:
        assert num < len(self._levels), f"Model has less levels ({self.num_levels}) than requested ({int})."
        return self._levels[num]

    @kwargs_decorator({"level_type": None,
                       "constraint": "soft"})
    def append_level(self, **kwargs) -> None:
        assert 'dim_output' in kwargs, f"No output dimension has been provided."

        # if no input dimension if specified, take the output of the previous level
        if "dim_input" not in kwargs and self.num_levels > 0:
            kwargs["dim_input"] = self.level(self.num_levels - 1).dim_output
        elif "dim_input" not in kwargs and not self.empty_sample:
            kwargs['dim_input'] = self.dim_input

        # overwrite param_trainable
        match kwargs['constraint']:
            case "soft":
                kwargs['param_trainable'] = True
            case "hard":
                kwargs['param_trainable'] = False
            case _:
                raise NameError('The constraint argument must be either soft or hard.')

        # create level and remove sample
        if 'sample' in kwargs:
            kwargs['sample'] = None
        level = factory(**kwargs)

        # verify dimension consistency
        if self.num_levels > 0:
            assert self.level(self.num_levels - 1).dim_output == level.dim_input, \
                f"The input dimension of the level to be added ({level.dim_input}) does not correspond to the " \
                f"output dimension of the previous level ({self.level(self.num_levels - 1).dim_output})."

        self._levels.append(level)

    def init_levels(self, full: bool = True) -> None:
        r"""
        Initializes all levels based on the model sample.

        :param full: If specified to False, the sample initialization is going to randomly initialized for the deeper
            levels to avoid running a forward on the full data. This is relevant in a stochastic setting. Defaults
            to True.
        :type full: bool, optional
        """

        if full:
            x = self.sample
            for level in self.levels:
                level.init_sample(sample=x)
                level.init_parameters(overwrite=False)
                x = level(x)
        else:
            for level in self.levels:
                level.num_sample = self.num_sample
                level.init_parameters(overwrite=False)

    def init_targets(self, targets: torch.Tensor, num_level: int = -1) -> None:
        try:
            self.level(num_level).targets = targets
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
