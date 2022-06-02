"""
Abstract RKM view class.

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
from .._sample import _sample


@utils.extend_docstring(_sample)
class view(_sample, metaclass=ABCMeta):
    r"""
    :param kernel: Initiates a view based on an existing kernel object. If the value is not `None`, all other
        parameters are neglected and inherited from the provided kernel., default to `None`
    :param bias: Bias
    :param bias_trainable: defaults to `False`
    :param dim_output: Output dimension

    :type kernel: rkm.kernel.base, optional
    :type bias: bool, optional
    :type bias_trainable: bool, optional
    :type dim_output: int, optional
    """

    @utils.kwargs_decorator({
        "kernel": None,
        "dim_output": None,
        "bias": False,
        "bias_trainable": False
    })
    def __init__(self, **kwargs):
        """
        A view is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(view, self).__init__(**kwargs)
        self._dim_output = kwargs["dim_output"]

        # BIAS
        self._bias_trainable = kwargs["bias_trainable"]
        self._requires_bias = kwargs["bias"]
        self._bias = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=self._bias_trainable)

        # KERNEL INIT
        # always share sample with kernel
        kernel = kwargs["kernel"]
        if kernel is None:
            self._kernel = factory(**{**kwargs,
                                      "sample": self.sample_as_param,
                                      "idx_sample": self.idx})
        elif isinstance(kernel, base):
            self._log.info("Initiating view based on existing kernel and overwriting its sample.")
            self._kernel = kernel
            self._kernel.init_sample(sample=self.sample_as_param,
                                     idx_sample=self.idx)
        else:
            raise TypeError("Argument kernel is not of the kernel class.")

        self._log.debug("View initialized with " + str(self._kernel))

    def __str__(self):
        return "View with " + str(self._kernel)

    def __repr__(self):
        return self.__str__()

    @property
    def bias(self):
        if self._bias.nelement() == 0:
            return None
        return self._bias.data.cpu().numpy()

    @bias.setter
    def bias(self, val):
        if val is not None:
            val = utils.castf(val, dev=self._bias.device).squeeze()
            dim_val = len(val.shape)

            # verifying the shape of the bias
            if dim_val == 0:
                val = val.repeat(self._dim_output)
            elif dim_val > 1:
                self._log.error("The bias can only be set to a scalar or a vector. "
                                "This operation is thus discarded.")
            # setting the value
            if self._bias.nelement() == 0:
                self._bias = torch.nn.Parameter(val,
                                                requires_grad=self._bias_trainable)
            else:
                self._bias.data = val
                # zeroing the gradients if relevant
                if self._bias_trainable:
                    self._bias.grad.data.zero_()

    @property
    def bias_trainable(self) -> bool:
        return self._bias_trainable

    @bias_trainable.setter
    def bias_trainable(self, val: bool):
        self._bias_trainable = val
        self._bias.requires_grad = self._bias_trainable

    @property
    def dim_output(self) -> int:
        r"""
        Output dimension
        """
        return self._dim_output

    @dim_output.setter
    def dim_output(self, val: int):
        self._log.error("This value cannot be set.")

    @property
    def kernel(self) -> base:
        r"""
        The kernel used by the model or view. Reassigning this value
        """
        return self._kernel

    @kernel.setter
    def kernel(self, val: base):
        self._kernel = val
        self.init_sample(self._kernel.sample_as_param)
        self._idx_sample = self._kernel.idx

    ## MATHS

    def phi(self, x=None) -> Tensor:
        return self.kernel.phi(x)

    def k(self, x=None) -> Tensor:
        return self._kernel.k(x)

    @abstractmethod
    def h(self, x=None) -> Tensor:
        pass

    @abstractmethod
    def w(self, x=None) -> Tensor:
        pass

    @abstractmethod
    def wh(self, x=None) -> Tensor:
        pass

    @abstractmethod
    def phiw(self, x=None) -> Tensor:
        pass

    @property
    def Phi(self) -> Tensor:
        return self.phi()

    @property
    def K(self) -> Tensor:
        return self._kernel.K

    @property
    def H(self) -> Tensor:
        return self.h()

    @property
    def W(self) -> Tensor:
        return self.w()

    @property
    def WH(self) -> Tensor:
        return self.wh()

    @property
    def PhiW(self) -> Tensor:
        return self.phiw()

    def forward(self, x=None):
        if x is None:
            return self.phiw(x) + self._bias[:, None]
