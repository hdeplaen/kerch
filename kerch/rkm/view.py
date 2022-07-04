"""
Abstract RKM View class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021, rewritten in June 2022
"""

import torch
from torch import Tensor
from math import sqrt

from kerch import utils
from kerch.kernel import factory, base
from kerch._sample import _Sample
from ._view import _View


@utils.extend_docstring(_Sample)
class View(_View, _Sample):
    r"""
    :param kernel: Initiates a View based on an existing kernel object. If the value is not `None`, all other
        parameters are neglected and inherited from the provided kernel., default to `None`
    :param bias: Bias
    :param bias_trainable: defaults to `False`

    :type kernel: kerpy.kernel.base, optional
    :type bias: bool, optional
    :type bias_trainable: bool, optional
    """

    @utils.kwargs_decorator({
        "kernel": None,
        "bias": False,
        "bias_trainable": False,
        "kappa": 1.
    })
    def __init__(self, *args, **kwargs):
        """
        A View is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(View, self).__init__(*args, **kwargs)

        # KAPPA
        self._kappa = kwargs["kappa"]
        self._kappa_sqrt = sqrt(self._kappa)

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
            self._log.info("Initiating View based on existing kernel and overwriting its sample.")
            self._kernel = kernel
            self._kernel.init_sample(sample=self.sample_as_param,
                                     idx_sample=self.idx)
        else:
            raise TypeError("Argument kernel is not of the kernel class.")

        self._log.debug("View initialized with " + str(self._kernel))

    def __str__(self):
        return "view with " + str(self._kernel)

    def _reset_hidden(self) -> None:
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                          requires_grad=self._hidden.requires_grad)

    def _reset_weight(self) -> None:
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                          requires_grad=self._weight.requires_grad)

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, val:float):
        self._kappa = val

    @property
    def bias(self) -> Tensor:
        if self._bias.nelement() != 0:
            return self._bias.data
        self._log.debug("No bias has been initialized yet.")

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
    def _bias_exists(self) -> bool:
        return self._bias.nelement() != 0

    ########################################################################
    @property
    def dim_feature(self) -> int:
        return self.kernel.dim_feature

    @property
    def kernel(self) -> base:
        r"""
        The kernel used by the model or View.
        """
        return self._kernel

    def set_kernel(self, val: base):
        r"""
        For some obscure reason, this does not work as a setter (@kernel.setter).
        TODO: find out why and solve
        """
        self._log.info("Updating View based on an external kernel and overwriting its sample.")
        self._kernel = val
        self._kernel.init_sample(sample=self.sample_as_param,
                                 idx_sample=self.idx)

    #########################################################################
    def update_hidden(self, val: Tensor, idx_sample=None) -> None:
        super(View, self).update_hidden(val, idx_sample)
        self._reset_weight()

    ## WEIGHT
    @property
    def weight(self) -> Tensor:
        return super(View, self).weight

    @property
    def weight_as_param(self) -> torch.nn.Parameter:
        if self._weight_exists:
            return self._weight
        self._log.debug("No weight has been initialized yet.")
        raise AttributeError

    @weight.setter
    def weight(self, val):
        if val is not None:
            # sets the parameter to an existing one
            if isinstance(val, torch.nn.Parameter):
                self._weight = val
            else:  # sets the value to a new one
                val = utils.castf(val, tensor=False, dev=self._weight.device)
                if self._weight_exists:
                    self._weight = torch.nn.Parameter(val, requires_grad=self._param_trainable)
                else:
                    self._weight.data = val
                    # zeroing the gradients if relevant
                    if self._param_trainable and self._weight.grad is not None:
                        self._weight.grad.data.zero_()

                self._dim_output = self._weight.shape[1]
            self._reset_hidden()
        else:
            self._log.info("The weight is unset.")

    @property
    def _weight_exists(self) -> bool:
        return self._weight.nelement() != 0

    def _update_weight_from_hidden(self):
        if self._hidden_exists:
            # will return a PrimalError if not available
            self.weight = self.Phi.T @ self.H
            self._log.debug("Setting the weight based on the hidden values.")
        else:
            self._log.info("The weight cannot based on the hidden values as these are unset.")

    ## MATHS

    def phi(self, x=None) -> Tensor:
        return self._kappa_sqrt * self.kernel.phi(x)

    def k(self, x=None) -> Tensor:
        return self.kappa * self._kernel.k(x)

    def h(self, x=None) -> Tensor:
        if x is None:
            if self._hidden_exists:
                return self._hidden[self.idx, :]
            else:
                self._log.warning("No hidden values exist or have been initialized.")
                raise utils.DualError(self)
        raise NotImplementedError

    def forward(self, x=None, representation="dual"):
        if self._bias_exists:
            return self.phiw(x, representation) + self._kappa_sqrt * self._bias[:, None]
        else:
            return self.phiw(x, representation)
