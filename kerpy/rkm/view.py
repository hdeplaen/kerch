"""
Abstract RKM view class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021, rewritten in June 2022
"""

import torch
from torch import Tensor

from kerpy import utils
from kerpy.kernel import factory, base
from kerpy._sample import _sample


@utils.extend_docstring(_sample)
class view(_sample):
    r"""
    :param kernel: Initiates a view based on an existing kernel object. If the value is not `None`, all other
        parameters are neglected and inherited from the provided kernel., default to `None`
    :param bias: Bias
    :param bias_trainable: defaults to `False`

    :type kernel: kerpy.kernel.base, optional
    :type bias: bool, optional
    :type bias_trainable: bool, optional
    """

    @utils.kwargs_decorator({
        "kernel": None,
        "dim_output": None,
        "bias": False,
        "bias_trainable": False,
        "hidden": None,
        "weight": None,
        "param_trainable": False
    })
    def __init__(self, **kwargs):
        """
        A view is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(view, self).__init__(**kwargs)
        self._dim_output = kwargs["dim_output"]

        # INITIATE
        self._param_trainable = kwargs["param_trainable"]
        weight = kwargs["weight"]
        hidden = kwargs["hidden"]
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._param_trainable)
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._param_trainable)
        if weight is not None and hidden is not None:
            self._log.info("Both the hidden and the weight are set. Priority is given to the hidden values.")
            self.hidden = hidden
        elif weight is None:
            self.hidden = hidden
        elif hidden is None:
            self.weight = weight

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

    def _reset_hidden(self) -> None:
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                          requires_grad=self._hidden.requires_grad)

    def _reset_weight(self) -> None:
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                          requires_grad=self._weight.requires_grad)

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

    @property
    def dim_output(self) -> int:
        r"""
        Output dimension
        """
        return self._dim_output

    @dim_output.setter
    def dim_output(self, val: int):
        self._log.error("This value cannot be set. Change the hidden property, or the weights "
                        "property if applicable.")

    @property
    def kernel(self) -> base:
        r"""
        The kernel used by the model or view.
        """
        return self._kernel

    def set_kernel(self, val: base):
        r"""
        For some obscure reason, this does not work as a setter (@kernel.setter).
        TODO: find out why and solve
        """
        self._log.info("Updating view based on an external kernel and overwriting its sample.")
        self._kernel = val
        self._kernel.init_sample(sample=self.sample_as_param,
                                 idx_sample=self.idx)

    ## HIDDEN

    @property
    def hidden(self) -> Tensor:
        if self._hidden_exists:
            return self._hidden.data[self.idx, :]

    def update_hidden(self, val: Tensor, idx_sample=None) -> None:
        # first verify the existence of the hidden values before updating them.
        if not self._hidden_exists:
            self._log.warning("Could not update hidden values as these do not exist. "
                              "Please set the values for hidden first.")
            return

        if idx_sample is None:
            idx_sample = self._all_sample()
        self._hidden.data[idx_sample, :] = val.data
        self._reset_weight()

    @property
    def hidden_as_param(self) -> torch.nn.Parameter:
        r"""
        The hidden values as a torch.nn.Parameter
        """
        if self._hidden_exists:
            return self._hidden
        self._log.debug("No hidden values have been initialized yet.")

    @hidden.setter
    def hidden(self, val):
        # sets the parameter to an existing one
        if val is not None:
            if isinstance(val, torch.nn.Parameter):
                self._hidden = val
            else:  # sets the value to a new one
                val = utils.castf(val, tensor=False, dev=self._hidden.device)
                if self._hidden_exists == 0:
                    self._hidden = torch.nn.Parameter(val, requires_grad=self._param_trainable)
                else:
                    self._hidden.data = val
                    # zeroing the gradients if relevant
                    if self._param_trainable:
                        self._hidden.grad.data.zero_()

                self._num_h, self._dim_output = self._hidden.shape
                self._reset_weight()
        else:
            self._log.info("The hidden value is unset.")

    @property
    def hidden_trainable(self) -> bool:
        return self._param_trainable

    @hidden_trainable.setter
    def hidden_trainable(self, val: bool):
        # changes the possibility of training the hidden values through backpropagation
        self._param_trainable = val
        self._hidden.requires_grad = self._param_trainable

    @property
    def _hidden_exists(self) -> bool:
        r"""
        Returns if this view has hidden variables attached to it.
        """
        return self._hidden.nelement() != 0

    ## WEIGHT
    @property
    def weight(self) -> Tensor:
        return self.weight_as_param.data

    @property
    def weight_as_param(self) -> torch.nn.Parameter:
        if self._weight_exists:
            return self._weight
        self._log.debug("No weight has been initialized yet.")

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
                    if self._param_trainable:
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
        return self.kernel.phi(x)

    def k(self, x=None) -> Tensor:
        return self._kernel.k(x)

    def h(self, x=None) -> Tensor:
        if x is None:
            if self._hidden_exists:
                return self._hidden[self.idx, :]
            else:
                self._log.warning("No hidden values exist or have been initialized.")
                raise utils.DualError(self)
        raise NotImplementedError

    def w(self, x=None) -> Tensor:
        if not self._weight_exists and self._hidden_exists:
            self._update_weight_from_hidden()

        if x is None:
            if self._weight_exists:
                return self._weight
            else:
                raise utils.PrimalError
        raise NotImplementedError

    def wh(self, x=None) -> Tensor:
        raise NotImplementedError

    def phiw(self, x=None, representation="dual") -> Tensor:
        def primal(x):
            return self.phi(x) @ self.W

        def dual(x):
            return self._kernel.k(x) @ self.H

        switcher = {"primal": primal,
                    "dual": dual}
        if representation in switcher:
            return switcher.get(representation)(x)
        else:
            raise utils.RepresentationError

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

    def forward(self, x=None, representation="dual"):
        if self._bias_exists:
            return self.phiw(x, representation) + self._bias[:, None]
        else:
            return self.phiw(x, representation)
