"""
Abstract RKM level class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from torch import Tensor

from abc import ABCMeta, abstractmethod

from .. import utils
from ..kernel import factory, base

class level(torch.nn.Module, metaclass=ABCMeta):
    @utils.kwargs_decorator({
        "_requires_bias": False,
        "kernel": None,
        "eta": 1.,
        "sample": None,
        "sample_trainable": False,
        "num_sample": 1.,
        "dim_input": None,
        "dim_output": None
    })
    def __init__(self, **kwargs):
        super(level, self).__init__()

        self._eta = kwargs["eta"]
        self._sample = kwargs["sample"]
        self._sample_trainable = kwargs["sample_trainable"]

        self._dim_input = kwargs["dim_input"]
        self._dim_output = kwargs["dim_output"]

        self._requires_bias = kwargs["_requires_bias"]

        # KERNEL INIT
        kernel = kwargs["kernel"]
        if kernel is None:
            self._kernel = factory(**kwargs)
            self.init_sample(self._kernel.sample_as_param)
        elif isinstance(kernel, base):
            self._kernel = kernel
            self.init_sample(self._kernel.sample_as_param)
            self.stochastic(self._kernel.idx)
        else:
            raise Exception("Argument kernel is not of the kernel class.")

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    @property
    def dim_input(self) -> int:
        assert self._dim_input is not None, "Input dimension not initialized yet."
        return self._dim_input

    @dim_input.setter
    def dim_input(self, val:int):
        raise NotImplementedError

    @property
    def dim_output(self) -> int:
        assert self._dim_output is not None, "Output dimension not initialized yet."
        return self._dim_output

    @dim_output.setter
    def dim_output(self, val:int):
        raise NotImplementedError

    @property
    def kernel(self) -> base:
        r"""
        The kernel used by the model or level.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, val:base):
        raise NotImplementedError

    @property
    def sample(self) -> Tensor:
        assert self._sample is not None, "Sample dataset has not been initialized yet."
        return self._sample

    @sample.setter
    def sample(self, val):
        raise NotImplementedError
        # val = utils.castf(val)

    @property
    def dim_sample(self) -> int:
        r"""
        Dimension of each datapoint.
        """
        return self._dim_sample

    @property
    def num_sample(self) -> int:
        r"""
        Number of datapoints in the sample set.
        """
        return self._sample.shape[0]

    @property
    def num_idx(self) -> int:
        r"""
        Number of selected datapoints of the sample set when performaing various operations. This is only relevant in
        the case of stochastic training.
        """
        return len(self._idx_sample)

    @property
    def _current_sample(self) -> Tensor:
        return self._sample[self._idx_sample, :]

    def _all_sample(self):
        return range(self.num_sample)

    def train(self, mode=True):
        r"""
        Sets the level in training mode, which disables the gradients computation and disables stochasticity of the
        kernel. For the gradients and other things, we refer to the `torch.nn.Module` documentation. For the stochastic
        part, when put in evaluation mode (`False`), all the sample points are used for the computations, regardless of
        the previously specified indices.
        """
        if not mode:
            self.stochastic()
        return self

    def stochastic(self, idx_sample=None, prop_sample=None):
        """
        Resets which subset of the samples are to be used until the next call of this function. This is relevant in the
        case of stochastic training.

        :param idx_sample: Indices of the sample subset relative to the original sample set., defaults to `None`
        :type idx_sample: int[], optional
        :param prop_sample: Instead of giving indices, passing a proportion of the original sample set is also
            possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
            such that :math:`0 <` `prop_sample` :math:`\leq 1`., defaults to `None`.
        :type prop_sample: double, optional

        If `None` is specified for both `idx_sample` and `prop_sample`, all samples are used and the subset equals the
        original sample set. This is also the default behavior if this function is never called, nor the parameters
        specified during initialization.

        .. note::
            Both `idx_sample` and `prop_sample` cannot be filled together as conflict would arise.
        """
        assert idx_sample is None or prop_sample is None, "Both idx_sample and prop_sample are not None. " \
                                                          "Please choose one non-None parameter only."

        if idx_sample is not None:
            self._idx_sample = idx_sample
        elif prop_sample is not None:
            assert prop_sample <= 1., 'Parameter prop_sample: the chosen proportion cannot be greater than 1.'
            assert prop_sample > 0., 'Parameter prop_sample: the chosen proportion must be strictly greater than 0.'
            n = self.num_sample
            k = torch.round(n * prop_sample)
            perm = torch.randperm(n)
            self._idx_sample = perm[:k]
        else:
            self._idx_sample = self._all_sample()

        self._kernel.stochastic(idx_sample=self._idx_sample)

    def init_sample(self, sample=None, idx_sample=None, prop_sample=None):
        r"""
        Initializes the sample set (and the stochastic indices).

        :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it
            will be given relative to these samples. In case of overwriting a current sample, `num_sample` and
            `dim_sample` are also overwritten. If `None` is specified, the sample dataset will be initialized according
            to `num_sample` and `dim_sample` specified during the construction. If a previous sample set has been used,
            it will keep the same dimension by consequence. A last case occurs when `sample` is of the class
            `torch.nn.Parameter`: the sample will then use those values and they can thus be shared with the module
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
            self._sample = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty((self._num_sample, self._dim_input), dtype=utils.FTYPE)),
                requires_grad=self._sample_trainable)
            self._kernel.init_sample(self._sample)
            self.stochastic(idx_sample, prop_sample)
        elif isinstance(sample, torch.nn.Parameter):
            self._num_sample, self._dim_input = sample.shape
            self._sample = sample
        else:
            self._num_sample, self._dim_input = sample.shape
            self._sample = torch.nn.Parameter(sample.data,
                                              requires_grad=self._sample_trainable)
            self._kernel.init_sample(self.sample)
            self.stochastic(idx_sample, prop_sample)

    def update_sample(self, sample_values, idx_sample=None):
        raise NotImplementedError

## MATHS

    @property
    def H(self):
        pass

    @property
    def V(self):
        pass

    @property
    def VH(self):
        pass

    @property
    def phi(self):
        pass

    @property
    def phiV(self):
        pass