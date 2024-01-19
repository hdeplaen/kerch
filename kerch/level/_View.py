"""
Abstract RKM View class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021, rewritten in June 2022
"""

import torch
from torch import Tensor
from abc import ABCMeta, abstractmethod
from typing import Union

from kerch import utils
from ..module.Stochastic import Stochastic


@utils.extend_docstring(Stochastic)
class _View(Stochastic, metaclass=ABCMeta):
    r"""
    :param dim_output: Output dimension. If None, it will later be assigned by the target is relevant. Defaults to None.
    :param representation: Specifies if the level works in primal or dual representation. The dual representation
        ('dual') is guaranteed to always have a finite dimensional formulation (the RKHS is finite dimensional as the
        sample has a finite number of datapoints). If the primal formulation ('primal') is available, the number of
        parameters (proportional to the property `dim_feature`) will potentially be smaller than for the dual
        (proportional to the number of sample datapoints, hence the formulation will be lighter hence faster.
        Defaults to 'dual'.
    :param weight: Weight values to start with. This is most of the cases not necessary if the level is meant to be
        trained based on a gr+adient or fitted. Defaults to None.
    :param hidden: Hidden values to start with. This is most of the cases not necessary if the level is meant to be
        trained based on a gradient or fitted. Defaults to None.
    :param_trainable: Boolean specifying whether the model parameters (weight, hidden and bias if applicable) are meant
        to have a gradient available. This is relevant is the level is to be trained based on a gradient. This is
        irrelevant if the model is meant to be fitted by a linear system. Defaults to False.

    :type dim_output: int, optional
    :type representation: str, optional
    :type weight: Tensor[dim_feature, dim_output], optional
    :type hidden: Tensor[num_sample, dim_output], optional
    :param_trainable: bool, optional
    """

    def __init__(self, *args, **kwargs):
        """
        A View is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(_View, self).__init__(*args, **kwargs)
        self._dim_output = kwargs.pop('dim_output', None)
        self._representation = utils.check_representation(kwargs.pop('representation', 'dual'), cls=self)

        weight = kwargs.pop('weight', None)
        hidden = kwargs.pop('hidden', None)

        # INITIATE
        self._param_trainable = kwargs.pop('param_trainable', False)
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

    @property
    def hparams(self) -> dict:
        return {'Output dimension': self.dim_output,
                'Representation': self.representation,
                'Parameters trainable': self._param_trainable,
                **super(_View, self).hparams}

    def _reset_hidden(self) -> None:
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE,
                                                      device=self._param_device),
                                          requires_grad=self._hidden.requires_grad)

    @abstractmethod
    def _reset_weight(self) -> None:
        pass

    def _init_hidden(self) -> None:
        assert self._num_total is not None, "No data has been initialized yet."
        assert self._dim_output is not None, "No output dimension has been provided."
        self.hidden = torch.nn.init.orthogonal_(torch.empty((self._num_total, self.dim_output),
                                                            dtype=utils.FTYPE, device=self._param_device))

    def _init_weight(self) -> None:
        assert self._dim_output is not None, "No output dimension has been provided."
        self.weight = torch.nn.init.orthogonal_(torch.empty((self.dim_feature, self.dim_output),
                                                            dtype=utils.FTYPE, device=self._param_device))

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

    @property
    def param_trainable(self) -> bool:
        r"""
        Specifies whether the parameters weight and hidden are trainable or not.
        """
        return self._param_trainable

    @param_trainable.setter
    def param_trainable(self, val: bool) -> None:
        self._hidden.requires_grad = val
        self._weight.requires_grad = val
        self._param_trainable = val

    @property
    def representation(self) -> str:
        return self._representation

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
        if self._dim_output is None:
            raise utils.NotInitializedError(cls=self, message="The output dimension has not been initialized yet.")
        return self._dim_output

    @dim_output.setter
    def dim_output(self, val: int) -> None:
        self._dim_output = int(val)
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
        raise utils.NotInitializedError(cls=self, message="No hidden values have been initialized or computed yet.")

    def update_hidden(self, val: Tensor, idx_sample=None) -> None:
        # first verify the existence of the hidden values before updating them.
        with torch.no_grad():
            if not self._hidden_exists:
                self._init_hidden()

            if idx_sample is None:
                idx_sample = self.idx
            self._hidden.copy_(val.data[idx_sample, :].T)
            if self._param_trainable and self._hidden.grad is not None:
                self._hidden.grad.sample.zero_()
            self._reset_weight()

    @hidden.setter
    def hidden(self, val):
        # sets the parameter to an existing one
        if val is not None:
            if isinstance(val, torch.nn.Parameter):
                del self._hidden
                self._hidden = val
                self._num_h, self._dim_output = self._hidden.shape
            else:  # sets the value to a new one
                with torch.no_grad():
                    # to work on the stiefel manifold, the parameters are required to have the number of components as
                    # first dimension
                    val = utils.castf(val, dev=self._param_device)
                    val = val.T
                    if self._hidden_exists and val.shape == self._hidden.shape:
                        self._hidden.copy_(val)
                        # zeroing the gradients if relevant
                        if self._param_trainable and self._hidden.grad is not None:
                            self._hidden.grad.sample.zero_()
                    else:
                        del self._hidden
                        # torch.no_grad() does not affect the constructor
                        self._hidden = torch.nn.Parameter(val, requires_grad=self._param_trainable)

                    self._dim_output, self._num_h = self._hidden.shape
                self._reset_weight()
        else:
            self._reset_hidden()
            self._reset_weight()
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

    @property
    def attached(self) -> bool:
        r"""
        Boolean indicating whether the view is attached to another multi_view.
        """
        try:
            return self._attached_weight is not None
        except AttributeError:
            return False

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
            raise utils.NotInitializedError(cls=self, message="No weight has been initialized or computed yet.")

    @weight.setter
    def weight(self, val):
        if self.attached:
            raise Exception("Cannot assign value of an attached view. Please change the value of the mother class "
                            "instead.")
        else:
            if val is not None:
                # sets the parameter to an existing one
                if isinstance(val, torch.nn.Parameter):
                    del self._weight
                    self._weight = val
                else:  # sets the value to a new one
                    with torch.no_grad():
                        val = utils.castf(val, tensor=False, dev=self._weight.device)
                        val = val.T
                        if self._weight_exists and self._weight.shape == val.shape:
                            self._weight.copy_(val.T)
                            # zeroing the gradients if relevant
                            if self._param_trainable and self._weight.grad is not None:
                                self._weight.grad.sample.zero_()
                        else:
                            del self._weight
                            # torch.no_grad() does not affect the constructor
                            self._weight = torch.nn.Parameter(val.T, requires_grad=self._param_trainable)

                        self._dim_output = self._weight.shape[0]
                self._reset_hidden()
            else:
                self._reset_weight()
                self._reset_hidden()
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

    @abstractmethod
    def _update_hidden_from_weight(self):
        pass

    @property
    def _param_device(self) -> torch.device:
        return self._hidden.device

    @abstractmethod
    def phi(self, x=None) -> Tensor:
        pass

    @abstractmethod
    def k(self, x=None) -> Tensor:
        pass

    def c(self, x=None) -> Tensor:
        phi = self.phi(x)
        return phi.T @ phi / self.num_idx

    def h(self, x=None) -> Union[Tensor, torch.nn.Parameter]:
        if not self._hidden_exists and self._weight_exists:
            try:
                self._update_hidden_from_weight()
            except NotImplementedError:
                self._log.info('The relation between the weights and the hidden variables have '
                               'not been implemented in this case.')

        if x is None:
            if self._hidden_exists:
                return self.hidden
            else:
                raise utils.ImplicitError(cls=self, message="No hidden values exist or have been initialized. "
                                                            "Please initialize the parameters or solve the model.")
        raise NotImplementedError

    def w(self, x=None) -> Union[Tensor, torch.nn.Parameter]:
        if not self.attached and not self._weight_exists and self._hidden_exists:
            self._update_weight_from_hidden()

        if x is None:
            return self.weight
        raise NotImplementedError

    def phiw(self, x=None, representation=None) -> Tensor:
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

    @property
    def hidden_correlation(self) -> Tensor:
        r"""
        Correlation of the hidden variables :math:`\mathbf{h}^\top \mathbf{h}`. This should be the identity provided
        that the hidden variables lie on the Stiefel manifold.
        """
        return self.hidden.T @ self.hidden

    @property
    def weight_correlation(self) -> Tensor:
        r"""
        Correlation of the weights :math:`\mathbf{w}^\top \mathbf{w}`. This should be the identity provided
        that the weights lie on the Stiefel manifold.
        """
        return self.weight.T @ self.weight

    @property
    def hidden_projector(self) -> Tensor:
        r"""
        Projector on the subspace spanned by the hidden variables :math:`\mathbf{h}\mathbf{h}^\top`.
        This is a rigorous projector provided its determinant is unity, e.g. when the hidden variables lie on the
        Stiefel manifold.
        """
        return self.hidden @ self.hidden.T

    @property
    def weight_projector(self) -> Tensor:
        r"""
        Projector on the subspace spanned by the weights :math:`\mathbf{w}\mathbf{w}^\top`.
        This is a rigorous projector provided its determinant is unity, e.g. when the weights lie on the Stiefel manifold.
        """
        return self.weight @ self.weight.T

    @abstractmethod
    def _forward(self, representation, x=None):
        pass

    def forward(self, x=None, representation=None):
        representation = utils.check_representation(representation, default=self._representation)
        name = f"forward_{id(x)}_{representation}"

        def get_level_key() -> str:
            if representation == self._representation:
                level_key = "forward_sample_default_representation" if x is None \
                    else "forward_oos_default_representation"
            else:
                level_key = "forward_sample_other_representation" if x is None \
                    else "forward_oos_other_representation"
            return level_key

        return self._get(name, level_key=lambda: get_level_key(), fun=lambda: self._forward(representation, x))
