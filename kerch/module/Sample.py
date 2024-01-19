# coding=utf-8
"""
Allows for a sample set manager, with stochastic support.
"""

from typing import Iterator
import torch

from abc import ABCMeta, abstractmethod
from torch import Tensor
from typing import Union

from kerch import utils
from .Cache import Cache
from .Logger import Logger
from .Stochastic import Stochastic
from ..transform import TransformTree
from ..utils import NotInitializedError


@utils.extend_docstring(Stochastic)
class Sample(Stochastic,  # manager stochastic indices
             Cache,  # creates a transportable cache (e.g. for GPU)
             Logger,  # allows logging actions and errors
             metaclass=ABCMeta):
    r"""
    :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it will
        be given relative to these samples., defaults to `None`
    :param sample_trainable: `True` if the gradients of the sample points are to be computed. If so, a graph is
        computed and the sample can be updated. `False` just leads to a static computation., defaults to `False`
    :param num_sample: Number of sample points. This parameter is neglected if `sample` is not `None` and overwritten by
        the number of points contained in sample., defaults to 1
    :param dim_input: Dimension of each sample point. This parameter is neglected if `sample` is not `None` and
        overwritten by the dimension of the sample points., defaults to 1

    :type sample: Tensor(num_sample, dim_input), optional
    :type sample_trainable: bool, optional
    :type num_sample: int, optional
    :type dim_input: int, optional

    :param idx_sample: Initializes the indices of the samples to be updated. All indices are considered if both
        `idx_stochastic` and `prop_stochastic` are `None`., defaults to `None`
    :param prop_sample: Instead of giving indices, specifying a proportion of the original sample set is also
        possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
        such that :math:`0 <` `prop_stochastic` :math:`\leq 1`. All indices are considered if both `idx_stochastic` and
        `prop_stochastic` are `None`., defaults to `None`.

    :type idx_sample: int[], optional
    :type idx_sample: float, optional

    :param sample_transform: TODO
    :type sample_transform: List[str]
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "sample": None,
        "sample_trainable": False,
        "num_sample": None,
        "dim_input": None,
        "idx_sample": None,
        "prop_sample": None,
        "sample_transform": []})
    def __init__(self, *args, **kwargs):
        super(Sample, self).__init__(*args, **kwargs)

        sample = kwargs["sample"]
        if sample is not None:
            sample = utils.castf(sample)
            self._num_total, self._dim_input = sample.shape
        else:
            self._dim_input = kwargs["dim_input"]
            self._num_total = kwargs["num_sample"]

        self._sample = torch.nn.Parameter(torch.empty((0, 0)))
        self._sample_trainable = kwargs["sample_trainable"]

        self.init_sample(sample, idx_sample=kwargs["idx_sample"], prop_sample=kwargs["prop_sample"])
        self._default_sample_transform = kwargs["sample_transform"]

    @property
    def dim_input(self) -> int:
        r"""
        Dimension of each datapoint.
        """
        if self._dim_input is None:
            raise NotInitializedError(cls=self, message="The input dimension has been initialized yet. "
                                                        "Please initialize it explicitly or assign a sample to "
                                                        "determine it.")
        return self._dim_input

    @dim_input.setter
    def dim_input(self, val: int):
        assert self._dim_input is None, "Cannot set the dimension of the sample points after initialization if the " \
                                        "sample data. Use init_sample() instead."
        self._dim_input = val
        if self._num_total is not None:
            self.init_sample()

    @property
    def empty_sample(self) -> bool:
        r"""
        Boolean specifying if the sample is empty or not.
        """
        return self._sample.nelement() == 0

    def _check_sample(self):
        if self.empty_sample:
            raise utils.NotInitializedError(cls=self, message='The sample has not been initialized yet.')

    @property
    def num_sample(self) -> int:
        r"""
        Number of datapoints in the sample set.
        """
        self._check_sample()
        return self._num_total

    @num_sample.setter
    def num_sample(self, val: int):
        assert self._num_total is None, "Cannot set the number of sample points after initialization of the " \
                                        "sample data. Use init_sample() instead."
        self._num_total = val
        if self._dim_input is not None:
            self.init_sample()

    @property
    def sample(self) -> torch.nn.Parameter:
        r"""
        Full original raw sample without any transform or potential stochastic selection.
        """
        self._check_sample()
        return self._sample

    @sample.setter
    def sample(self, val):
        assert val is not None, "Cannot assign the sample to None. Please use init_sample() if you want to " \
                                "re-initialize it."
        self.init_sample(val)

    @property
    def sample_trainable(self) -> bool:
        r"""
        Boolean if the sample data can be trained.
        """
        return self._sample_trainable

    @sample_trainable.setter
    def sample_trainable(self, val: bool):
        self._sample_trainable = val
        self._sample.requires_grad = self._sample_trainable

    @property
    def current_sample_projected(self) -> Tensor:
        r"""
        Returns the sample that is currently used in the computations and for the normalizing and centering statistics
        if relevant.
        """
        return self.sample_transform.projected_sample

    @property
    def current_sample(self) -> Tensor:
        return self._sample[self.idx, :]

    @torch.no_grad()
    def init_sample(self, sample=None, idx_sample=None, prop_sample=None):
        r"""
        Initializes the sample set (and the stochastic indices).

        :param sample: Sample points used for the various computations. When an out-of-sample computation is asked, it
            will be given relative to these samples. In case of overwriting a current sample, `num_sample` and
            `dim_input` are also overwritten. If `None` is specified, the sample data will be initialized according
            to `num_sample` and `dim_input` specified during the construction. If a previous sample set has been used,
            it will keep the same dimension by consequence. A last case occurs when `sample` is of the class
            `torch.nn.Parameter`: the sample will then use those values, and they can thus be shared with the level
            calling this method., defaults to `None`
        :type sample: Tensor, optional
        :param idx_sample: Initializes the indices of the samples to be updated. All indices are considered if both
            `idx_sample` and `prop_sample` are `None`., defaults to `None`
        :type idx_sample: int[], optional
        :param prop_sample: Instead of giving indices, specifying a proportion of the original sample set is also
            possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
            such that :math:`0 <` `prop_sample` :math:`\leq 1`. All indices are considered if both `idx_sample` and
            `prop_sample` are `None`., defaults to `None`.
        """
        sample = utils.castf(sample)

        if sample is None:
            self._log.debug("Initializing new sample with the sample dimensions.")
            if self._num_total is None or self.dim_input is None:
                self._log.info(
                    'The sample cannot be initialized because no sample data has been provided nor the '
                    'sample dimensions have been initialized yet.')
                return
            self._sample = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty((self._num_total, self._dim_input),
                                                      dtype=utils.FTYPE,
                                                      device=self._sample.device)),
                requires_grad=self._sample_trainable)
        elif isinstance(sample, torch.nn.Parameter):
            self._log.debug("Using existing sample defined as an external parameter.")
            self._num_total, self._dim_input = sample.shape
            self._sample_trainable = sample.requires_grad
            self._sample = sample
        else:
            self._log.debug("Assigning new sample points based on the given values.")
            self._num_total, self._dim_input = sample.shape
            self._sample = torch.nn.Parameter(sample.data,
                                              requires_grad=self._sample_trainable)

        self.stochastic(idx=idx_sample, prop=prop_sample)
        self._reset_cache(reset_persisting=False)

    @torch.no_grad()
    def update_sample(self, sample_values, idx_sample=None):
        r"""
        Updates the sample set. In contradiction to `init_samples`, this only updates the values of the sample and sets
        the gradients of the updated values to zero if relevant.

        :param sample_values: Values given to the updated samples.
        :param idx_sample: Indices of the samples to be updated. All indices are considered if `None`., defaults to
            `None`
        :type sample_values: Tensor
        :type idx_sample: int[], optional
        """

        if self.empty_sample:
            self._log.warning('Cannot update the sample values of a None sample data.')
            return

        sample_values = utils.castf(sample_values, dev=self._sample.device)

        # use all indices if unspecified
        if idx_sample is None:
            idx_sample = self._all_idx

        # check consistency of indices
        assert len(idx_sample) == sample_values.shape[0], f"Number of sample values ({sample_values.shape[0]}) and " \
                                                          f"corresponding indices ({len(idx_sample)}) are not " \
                                                          f"consistent."
        # update the values
        self._sample.data[idx_sample, :] = sample_values.data

        # zeroing relevant gradient if relevant
        if self._sample_trainable:
            self._sample.grad.data[idx_sample, :].zero_()

        self._reset_cache(reset_persisting=False)

    def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        if not self.empty_sample:
            yield self._sample
        yield from super(Sample, self)._euclidean_parameters(recurse)

    @property
    def sample_transform(self) -> TransformTree:
        r"""
        Default transform used by the sample.
        """
        return self._sample_transform

    @property
    def _sample_transform(self) -> TransformTree:
        def fun():
            return TransformTree(explicit=True,
                                 sample=self._sample[self.idx, :],
                                 default_transform=self._default_sample_transform,
                                 cache_level=self._cache_level)

        return self._get("sample_transform", level_key="sample_transform", fun=fun)

    def project_input(self, data) -> Union[Tensor, None]:
        r"""
        Apply to value the same transform as on the sample.
        """
        if data is None:
            return None
        return self.sample_transform.apply(data)

    def transform_sample_revert(self, data) -> Tensor:
        r"""
        Get back the original value from a projected value, by using the same transform as the sample,
        but in reverse. This is not always feasible, depending on the transform used (normalizations are
        typically not invertible as they are transform which are not bijective).
        """
        return self.sample_transform.revert(data)

    @property
    def hparams(self) -> dict:
        return {'Input dimension': self.input_dim,
                'Trainable sample': self._sample_trainable,
                'Default sample transforms': self._default_sample_transform,
                **super(Sample, self).hparams}

    @property
    def params(self) -> dict:
        return {**super(Sample, self).params}
