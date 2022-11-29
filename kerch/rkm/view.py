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

import kerch.rkm
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
        "kappa": 1.,
        "targets" : None
    })
    def __init__(self, *args, **kwargs):
        """
        A View is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(View, self).__init__(*args, **kwargs)

        # KAPPA
        self._kappa = kwargs["kappa"]
        self._kappa_sqrt = sqrt(self._kappa)
        self._mask = None

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
                                      "sample": self.sample,
                                      "idx_sample": self.idx})
        elif isinstance(kernel, base):
            self._log.info("Initiating View based on existing kernel and overwriting its sample.")
            self._kernel = kernel
            self._kernel.init_sample(sample=self.sample,
                                     idx_sample=self.idx)
        else:
            raise TypeError("Argument kernel is not of the kernel class.")

        # TARGETS
        targets = kwargs["targets"]
        targets = utils.castf(targets, tensor=True)
        if self._dim_output is None and targets is not None:
            self._dim_output = targets.shape[1]
        self.targets = targets



        self._log.debug("View initialized with " + str(self._kernel))

    def __str__(self):
        return "view with " + str(self._kernel)

    def _reset_weight(self) -> None:
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE,
                                                      device=self._weight.device),
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
            return self._bias
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

    ####################################################################################################################
    ## TARGETS

    @property
    def targets(self) -> Tensor:
        r"""
            Targets to be matched to.
        """
        if self._targets is None:
            self._log.warning("Empty target values.")
        return self._targets

    @targets.setter
    def targets(self, val):
        val = utils.castf(val, dev=self._sample.device, tensor=True)
        if val is None:
            self._log.debug("Targets set to empty values.")
        else:
            assert self.dim_output == val.shape[1], f"The shape of the given target {val.shape[1]} does not match the" \
                                                    f" required one {self.dim_output}."
            assert self.num_sample == val.shape[0], f"The number of target points {val.shape[0]} does not match the " \
                                                    f"required one {self.dim_input}."
        self._targets = torch.nn.Parameter(val)

    @property
    def current_targets(self) -> Tensor:
        r"""
            Returns the targets that are currently used in the computations and for the normalizing and centering
            statistics if relevant.
        """
        return self.targets[self.idx, :]


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
        self._kernel.init_sample(sample=self.sample,
                                 idx_sample=self.idx)

    #########################################################################
    ## WEIGHT
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
        return self.kappa * self.kernel.k(x)

    @property
    def K(self) -> Tensor:
        return self.kappa * self.kernel.K

    def forward(self, x=None, representation=None):
        representation = utils.check_representation(representation, default=self._representation)
        if self._bias_exists:
            return self.phiw(x, representation) + self._kappa_sqrt * self._bias[:, None]
        else:
            return self.phiw(x, representation)
