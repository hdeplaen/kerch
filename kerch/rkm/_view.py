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
from typing import Union

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
        "param_trainable": True
    })
    def __init__(self, *args, **kwargs):
        """
        A View is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(_View, self).__init__(*args, **kwargs)
        self._dim_output = kwargs["dim_output"]

        weight = kwargs["weight"]
        hidden = kwargs["hidden"]

        # INITIATE
        self._param_trainable = kwargs["param_trainable"]
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._param_trainable)
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._param_trainable)
        if weight is not None and hidden is not None:
            self._log.info("Both the hidden and the weight are set. Priority is given to the weight values.")
            self.weight = weight
        elif weight is None:
            self.hidden = hidden
        elif hidden is None:
            self.weight = weight

        self._attached_weight = None

    def _reset_hidden(self) -> None:
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE,
                                                      device=self._hidden.device),
                                          requires_grad=self._hidden.requires_grad)

    @abstractmethod
    def _reset_weight(self) -> None:
        pass

    def _init_hidden(self) -> None:
        assert self._num_total is not None, "No dataset has been initialized yet."
        assert self._dim_output is not None, "No output dimension has been provided."
        self.hidden = torch.nn.init.orthogonal_(torch.empty((self._num_total, self.dim_output),
                                                            dtype=utils.FTYPE, device=self._hidden.device))

    def _init_weight(self) -> None:
        assert self._dim_output is not None, "No output dimension has been provided."
        self.weight = torch.nn.init.orthogonal_(torch.empty((self.dim_feature, self.dim_output),
                                                            dtype=utils.FTYPE, device=self._weight.device))

    def init_parameters(self, representation=None, overwrite=True) -> None:
        """
        Initializes the model parameters: the weight in primal and the hidden values in dual.
        This is suitable for gradient-based training.

        :param representation: 'primal' or 'dual'
        :param overwrite: Does not initialize already initialized parameters if False., defaults to True
        :type representation: str, optional
        :type overwrite: bool, optional
        """
        representation = utils.check_representation(representation, self._representation, cls=self)

        def init_weight():
            if overwrite or not self._weight_exists:
                self._init_weight()

        def init_hidden():
            if overwrite or not self._hidden_exists:
                self._init_hidden()

        switcher = {"primal": init_weight,
                    "dual": init_hidden}
        switcher.get(representation)()

    ################################################################"
    @property
    @abstractmethod
    def dim_feature(self) -> int:
        """
        Dimension of the explicit feature map if relevant.
        """
        pass

    @property
    def dim_output(self) -> int:
        r"""
        Output dimension.
        """
        return self._dim_output

    @dim_output.setter
    def dim_output(self, val: int):
        self._dim_output = val
        self._reset_weight()
        self._reset_hidden()

    ##################################################################################################################
    ## HIDDEN

    @property
    def hidden(self) -> Tensor:
        """
        Hidden values.
        """
        if self._hidden_exists:
            return self._hidden.T[self.idx, :]
        self._log.debug("No hidden values have been initialized yet.")

    def update_hidden(self, val: Tensor, idx_sample=None) -> None:
        # first verify the existence of the hidden values before updating them.
        if not self._hidden_exists:
            self._log.warning("Could not update hidden values as these do not exist. "
                              "Please set the values for hidden first.")
            return

        if idx_sample is None:
            idx_sample = self._all_sample()
        self._hidden.data.T[idx_sample, :] = val.data
        self._reset_weight()

    @hidden.setter
    def hidden(self, val):
        # sets the parameter to an existing one
        if val is not None:
            if isinstance(val, torch.nn.Parameter):
                self._hidden = val
                self._num_h, self._dim_output = self._hidden.shape
            else:  # sets the value to a new one
                # to work on the stiefel manifold, the parameters are required to have the number of components as
                # as first dimension
                val = utils.castf(val, tensor=True, dev=self._hidden.device)
                self._hidden.data = val.T

                # zeroing the gradients if relevant
                if self._param_trainable and self._hidden.grad is not None:
                    self._hidden.grad.data.zero_()

                self._dim_output, self._num_h = self._hidden.shape
                self._reset_weight()
        else:
            self._log.info("The hidden value is unset.")

    @property
    def hidden_trainable(self) -> bool:
        """
        Returns whether the hidden variables are trainable (a gradient can be computed on it).
        """
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

    ####################################################################################################################
    ## ATTACH
    @property
    def attached(self) -> bool:
        r"""
        Boolean indicating whether the view is attached to another multiview.
        """
        return self._attached_weight is not None

    def attach_to(self, weight_fn) -> None:
        self._log.debug(self.__repr__() + " is attached to a multi-view.")
        assert not self.attached, 'Cannot attach a view which is already attached'
        self._attached_weight = weight_fn

    def detach(self) -> None:
        self._log.debug(self.__repr__() + ' is now detached.')
        w = self.weight
        self._attached = False
        self.weight = w

    ## WEIGHT
    @property
    def weight(self) -> Tensor:
        r"""
        Weight.
        """
        if self.attached:
            return self._attached_weight()
        else:
            if self._weight_exists:
                return self._weight.T
            self._log.debug("No weight has been initialized yet.")

    @weight.setter
    def weight(self, val):
        if self.attached:
            raise Exception("Cannot assign value of an attached view. Please change the value of the mother class "
                            "instead.")
        else:
            if val is not None:
                # sets the parameter to an existing one
                if isinstance(val, torch.nn.Parameter):
                    self._weight = val
                else:  # sets the value to a new one
                    val = utils.castf(val, tensor=False, dev=self._weight.device)
                    if self._weight_exists:
                        self._weight.data = val.T
                        # zeroing the gradients if relevant
                        if self._param_trainable and self._weight.grad is not None:
                            self._weight.grad.data.zero_()
                    else:
                        self._weight = torch.nn.Parameter(val.T, requires_grad=self._param_trainable)

                    self._dim_output = self._weight.shape[0]
                self._reset_hidden()
            else:
                self._log.info("The weight is unset.")

    @property
    def _weight_exists(self) -> bool:
        try:
            return self._weight.nelement() != 0
        except AttributeError:
            return False

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

    def h(self, x=None) -> Union[Tensor, torch.nn.Parameter]:
        if x is None:
            if self._hidden_exists:
                return self.hidden
            else:
                self._log.warning("No hidden values exist or have been initialized.")
                raise utils.DualError(self)
        raise NotImplementedError

    def w(self, x=None) -> Union[Tensor, torch.nn.Parameter]:
        if not self._weight_exists and self._hidden_exists:
            self._update_weight_from_hidden()

        if x is None:
            if self._weight_exists:
                return self.weight
            else:
                raise utils.PrimalError
        raise NotImplementedError

    def phiw(self, x=None, representation="dual") -> Tensor:
        def primal(x):
            return self.phi(x) @ self.W

        def dual(x):
            return self.kernel.k(x) @ self.H

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
    @abstractmethod
    def K(self) -> Tensor:
        pass

    @property
    def C(self) -> Tensor:
        return self.c()

    @property
    def H(self) -> Union[Tensor, torch.nn.Parameter]:
        return self.h()

    @property
    def W(self) -> Union[Tensor, torch.nn.Parameter]:
        return self.w()

    @property
    def PhiW(self) -> Tensor:
        return self.phiw()

    @abstractmethod
    def forward(self, x=None, representation="dual"):
        pass
