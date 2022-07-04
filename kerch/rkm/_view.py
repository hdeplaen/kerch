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
from abc import ABCMeta, abstractmethod

from kerch import utils
from kerch.kernel import factory, base
from kerch._stochastic import _Stochastic


@utils.extend_docstring(_Stochastic)
class _View(_Stochastic, metaclass=ABCMeta):
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
        "dim_output": None,
        "hidden": None,
        "weight": None,
        "param_trainable": True,
    })
    def __init__(self, *args, **kwargs):
        """
        A View is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(_View, self).__init__(*args, **kwargs)
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

    @abstractmethod
    def _reset_hidden(self) -> None:
        pass

    @abstractmethod
    def _reset_weight(self) -> None:
        pass

    ################################################################"
    @property
    @abstractmethod
    def dim_feature(self) -> int:
        pass

    @property
    def dim_output(self) -> int:
        r"""
        Output dimension
        """
        return self._dim_output

    @dim_output.setter
    def dim_output(self, val: int):
        self._dim_output = val
        self._reset_weight()
        self._reset_hidden()

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

    ##################################################################################################################
    ## HIDDEN

    @property
    def hidden(self) -> Tensor:
        if self._hidden_exists:
            return self._hidden.data[self.idx, :]
        raise AttributeError

    def update_hidden(self, val: Tensor, idx_sample=None) -> None:
        # first verify the existence of the hidden values before updating them.
        if not self._hidden_exists:
            self._log.warning("Could not update hidden values as these do not exist. "
                              "Please set the values for hidden first.")
            return

        if idx_sample is None:
            idx_sample = self._all_sample()
        self._hidden.data[idx_sample, :] = val.data

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
        Returns if this View has hidden variables attached to it.
        """
        return self._hidden.nelement() != 0

    ## WEIGHT
    @property
    def weight(self) -> Tensor:
        return self.weight_as_param.data

    @property
    @abstractmethod
    def weight_as_param(self) -> torch.nn.Parameter:
        pass

    @weight.setter
    @abstractmethod
    def weight(self, val):
        pass

    @property
    @abstractmethod
    def _weight_exists(self) -> bool:
        pass

    @abstractmethod
    def _update_weight_from_hidden(self):
        pass

    ## MATHS

    @abstractmethod
    def phi(self, x=None) -> Tensor:
        pass

    @abstractmethod
    def k(self, x=None) -> Tensor:
        pass

    def c(self, x=None) -> Tensor:
        phi = self.phi(x)
        return phi.T @ phi

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
        return self.kappa * self._kernel.K

    @property
    def C(self) -> Tensor:
        return self.c()

    @property
    def H(self) -> Tensor:
        return self.h()

    @property
    def W(self) -> Tensor:
        return self.w()

    @property
    def PhiW(self) -> Tensor:
        return self.phiw()

    @abstractmethod
    def forward(self, x=None, representation="dual"):
        pass
