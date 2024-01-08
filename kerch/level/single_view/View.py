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

from .._View import _View
from ...kernel._Kernel import _Kernel
from ...kernel.factory import class_factory
from ...utils import DEFAULT_KERNEL_TYPE, extend_docstring, kwargs_decorator, check_representation, \
    castf, NotInitializedError, FTYPE


@extend_docstring(_View)
@extend_docstring(_Kernel)
class View(_Kernel, _View):
    r"""
    :param kernel_type: Represents which kernel to use if kernel_class is not specified.
        Defaults to kerch.DEFAULT_KERNEL_TYPE.
    :param kernel_class: Instead of specifying the kernel type (which is restricted to the kernels implemented by
        default), a specific class can be specified here. This is relevant for example in the case of a
        self-implemented kernel. It must however inherit from kerch.kernel._Kernel. If not specified, the kernel_type
        argument is used to specify the kernel.
    :param bias: Bias
    :param bias_trainable: defaults to `False`

    :type bias: bool, optional
    :type bias_trainable: bool, optional
    :type kernel_type: str, optional
    :type kernel_class: kerch.kernel._Kernel, optional
    """
    def __new__(cls, *args, **kwargs):
        kernel_type = kwargs.pop('kernel_type', DEFAULT_KERNEL_TYPE)
        kernel_class = kwargs.pop('kernel_class', None)

        if kernel_class is None:
            kernel_class = class_factory(kernel_type)
        assert issubclass(kernel_class, _Kernel), 'The provided kernel_class argument is not a valid kernel class.'
        new_cls = type(cls.__name__, (cls, kernel_class, ), dict(cls.__dict__))
        return object.__new__(new_cls)

    @kwargs_decorator({
        # "kernel": None,
    })
    def __init__(self, *args, **kwargs):
        """
        A View is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(View, self).__init__(*args, **kwargs)

        # KAPPA
        self._kappa = kwargs.pop("kappa", 1.)
        self._kappa_sqrt = sqrt(self._kappa)
        self._mask = None

        bias = kwargs.pop('bias', None)

        # BIAS
        self._bias_trainable = kwargs.pop('bias_trainable', False)
        self._requires_bias = kwargs.pop('requires_bias', False)
        self._bias = torch.nn.Parameter(torch.empty(0, dtype=FTYPE),
                                        requires_grad=self._requires_bias and self._param_trainable)
        if bias is not None and self._requires_bias:
            self.bias = bias

        # target
        self._target = None
        target = kwargs.pop('target', None)
        target = castf(target)
        if self._dim_output is None and target is not None:
            self._dim_output = target.shape[1]
        self.target = target

        self._log.debug("View initialized with " + self.kernel.__str__())

    def __str__(self):
        if self.attached:
            return "view with " + self.kernel.__str__()
        return self.kernel.__str__()

    def init_parameters(self, representation=None, overwrite=True) -> None:
        super(View, self).init_parameters(representation=representation, overwrite=overwrite)
        if self.requires_bias and (not self._bias_exists or overwrite):
            self._init_bias()

    def _init_bias(self):
        assert self._num_total is not None, "No data has been initialized yet."
        assert self._dim_output is not None, "No output dimension has been provided."
        self.bias = torch.zeros((self.dim_output), dtype=FTYPE, device=self._bias.device)

    @property
    def _bias_exists(self) -> bool:
        return self._bias.nelement() != 0

    @property
    def requires_bias(self) -> bool:
        return self._requires_bias

    @requires_bias.setter
    def requires_bias(self, val: bool):
        self._requires_bias = val
        self._bias.requires_grad = val and self._param_trainable

    def _reset_weight(self) -> None:
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=FTYPE,
                                                      device=self._weight.device),
                                          requires_grad=self._weight.requires_grad)

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, val: float):
        self._kappa = val

    @property
    def bias(self) -> Tensor:
        if self._bias.nelement() != 0:
            return self._bias
        self._log.debug("No bias has been initialized yet.")

    @bias.setter
    @torch.no_grad()
    def bias(self, val):
        if val is not None:
            val = castf(val, dev=self._bias.device).squeeze()
            dim_val = len(val.shape)

            # verifying the shape of the bias
            if dim_val == 0:
                val = val.repeat(self._dim_output)
            elif dim_val > 1:
                self._log.error("The bias can only be set to a scalar or a vector. "
                                "This operation is thus discarded.")
            # setting the value
            if self._bias.nelement() == 0:
                self._bias = torch.nn.Parameter(val, requires_grad=self.bias_trainable)
            else:
                self._bias.data = val
                # zeroing the gradients if relevant
                if self._bias_trainable:
                    self._bias.grad.sample.zero_()

    @property
    def bias_trainable(self) -> bool:
        return self._requires_bias and self._param_trainable

    @property
    def param_trainable(self) -> bool:
        r"""
        Specifies whether the parameters weight and hidden are trainable or not.
        """
        return self._param_trainable

    @param_trainable.setter
    def param_trainable(self, val: bool) -> None:
        self._param_trainable = val
        self._hidden.requires_grad = val
        self._weight.requires_grad = val
        self._bias.requires_grad = self.bias_trainable


    @property
    def dim_feature(self) -> int:
        return self.kernel.dim_feature

    @property
    def target(self) -> Tensor:
        r"""
        target to be matched to.
        """
        if self._target is None:
            raise NotInitializedError(cls=self, message="The target values have not been set (yet).")
        return self._target

    @target.setter
    def target(self, val):
        val = castf(val, dev=self._sample.device, tensor=True)
        if val is None:
            self._log.debug("target set to empty values.")
        else:
            if self.empty_sample:
                raise NotInitializedError(cls=self, message="The sample has not been initialized yet.")
            assert self.dim_output == val.shape[
                1], f"The shape of the given target {val.shape[1]} does not match the" \
                    f" required one {self.dim_output}."
            assert self.num_sample == val.shape[
                0], f"The number of target points {val.shape[0]} does not match the " \
                    f"required one {self.num_sample}."
            self._target = torch.nn.Parameter(val, requires_grad=False)

    @property
    def current_target(self) -> Tensor:
        r"""
        Returns the target that are currently used in the computations, taking the stochastic aspect into account
        if relevant.
        """
        return self.target[self.idx, :]

    def _update_weight_from_hidden(self):
        if self._hidden_exists:
            # will return a ExplicitError if not available
            self.weight = self.Phi.T @ self.H
            self._log.debug("Setting the weight _Based on the hidden values.")
        else:
            self._log.info("The weight cannot _Based on the hidden values as these are unset.")

    def _update_hidden_from_weight(self):
        raise NotImplementedError

    ## MATHS
    def phi(self, x=None, projections=None) -> Tensor:
        return self._kappa_sqrt * self.kernel.phi(x, projections)

    def k(self, x=None, y=None, explicit=None, projections=None) -> Tensor:
        return self.kappa * self.kernel.k(x, y, explicit, projections)

    @property
    def K(self) -> Tensor:
        return self.kappa * self.kernel.K

    def _forward(self, representation, x=None):
        if self.requires_bias:
            return self.phiw(x, representation) + self._kappa_sqrt * self._bias[None, :]
        else:
            return self.phiw(x, representation)

    def forward(self, x=None, representation=None) -> Tensor:
        return _View.forward(self, x, representation)

    @property
    def kernel(self) -> _Kernel:
        return super(View, self)