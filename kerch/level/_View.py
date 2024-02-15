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
from ..feature.stochastic import Stochastic


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
        trained based on a gradient or fitted. Defaults to None.
    :param hidden: Hidden values to start with. This is most of the cases not necessary if the level is meant to be
        trained based on a gradient or fitted. Defaults to None.
    :param_trainable: Boolean specifying whether the model parameters (weight, hidden and bias if applicable) are meant
        to have a gradient available. This is relevant is the level is to be trained based on a gradient. This is
        irrelevant if the model is meant to be fitted by a linear system. Defaults to False.

    :type dim_output: int, optional
    :type representation: str, optional
    :type weight: torch.Tensor [dim_feature, dim_output], optional
    :type hidden: torch.Tensor [num_sample, dim_output], optional
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
        self._level_trainable = kwargs.pop('level_trainable', False)
        self._dual_param = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._level_trainable)
        self._primal_param = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._level_trainable)
        if weight is not None and hidden is not None:
            self._logger.info("Both the hidden and the weight are set. Priority is given to the weight values.")
            self.primal_param = weight
        elif weight is None:
            self.dual_param = hidden
        elif hidden is None:
            self.primal_param = weight

        self._attached_primal_param = None

    @property
    def hparams_fixed(self) -> dict:
        constraint = 'soft' if self._level_trainable else 'hard'
        return {'Output dimension': self.dim_output,
                'Representation': self.representation,
                'Parameters trainable': self._level_trainable,
                'Constraint': constraint,
                **super(_View, self).hparams_fixed}

    def _reset_dual(self) -> None:
        self._dual_param = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE,
                                                          device=self._param_device),
                                              requires_grad=self._dual_param.requires_grad)

    @abstractmethod
    def _reset_primal(self) -> None:
        pass

    def _init_dual(self) -> None:
        assert self._num_total is not None, "No data has been initialized yet."
        assert self._dim_output is not None, "No output dimension has been provided."
        self.dual_param = torch.nn.init.orthogonal_(torch.empty((self._num_total, self.dim_output),
                                                                dtype=utils.FTYPE, device=self._param_device))
        self._logger.info('The hidden variables are initialized.')

    def _init_primal(self) -> None:
        assert self._dim_output is not None, "No output dimension has been provided."
        self.primal_param = torch.nn.init.orthogonal_(torch.empty((self.dim_feature, self.dim_output),
                                                                  dtype=utils.FTYPE, device=self._param_device))
        self._logger.info('The weights are initialized.')

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
            if overwrite or not self._primal_param_exists:
                self._init_primal()

        def init_hidden():
            if overwrite or not self._dual_param_exists:
                self._init_dual()

        switcher = {"primal": init_weight,
                    "dual": init_hidden}
        switcher.get(representation)()

    @property
    def level_trainable(self) -> bool:
        r"""
        Specifies whether the parameters weight and hidden are trainable or not.
        """
        return self._level_trainable

    @level_trainable.setter
    def level_trainable(self, val: bool) -> None:
        self._dual_param.requires_grad = val
        self._primal_param.requires_grad = val
        self._level_trainable = val

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
        self._reset_primal()
        self._reset_dual()

    ##################################################################################################################
    ## HIDDEN

    @property
    def dual_param(self) -> Tensor:
        """
        Dual parameter of size [num_idx, dim_output].
        """
        if self._dual_param_exists:
            return self._dual_param.T[self.idx, :]
        raise utils.NotInitializedError(cls=self, message="No hidden values have been initialized or computed yet.")

    def update_dual(self, val: Tensor, idx_sample=None) -> None:
        # first verify the existence of the hidden values before updating them.
        with torch.no_grad():
            if not self._dual_param_exists:
                self._init_dual()

            if idx_sample is None:
                idx_sample = self.idx
            self._dual_param.copy_(val.data[idx_sample, :].T)
            if self._level_trainable and self._dual_param.grad is not None:
                self._dual_param.grad.zero_()
            self._reset_primal()

    @dual_param.setter
    def dual_param(self, val):
        # sets the parameter to an existing one
        if val is not None:
            with torch.no_grad():
                # to work on the stiefel manifold, the parameters are required to have the number of components as
                # first dimension
                val = utils.castf(val, dev=self._param_device)
                val = val.T
                if self._dual_param_exists and val.shape == self._dual_param.shape:
                    self._dual_param.copy_(val)
                    # zeroing the gradients if relevant
                    if self._level_trainable and self._dual_param.grad is not None:
                        self._dual_param.grad.zero_()
                else:
                    del self._dual_param
                    # torch.no_grad() does not affect the constructor
                    self._dual_param = torch.nn.Parameter(val, requires_grad=self._level_trainable)

                self._dim_output, self._num_h = self._dual_param.shape
            self._reset_primal()
        else:
            self._reset_dual()
            self._reset_primal()
            self._logger.info("The hidden value is unset.")

    @property
    def dual_trainable(self) -> bool:
        """
        Returns whether the hidden variables are trainable (a gradient can be computed on it).
        """
        return self._level_trainable

    @dual_trainable.setter
    def dual_trainable(self, val: bool):
        # changes the possibility of training the hidden values through backpropagation
        self._level_trainable = val
        self._dual_param.requires_grad = self._level_trainable

    @property
    def _dual_param_exists(self) -> bool:
        r"""
        Returns if this View has hidden variables attached to it.
        """
        return self._dual_param.nelement() != 0

    @property
    def attached(self) -> bool:
        r"""
        Boolean indicating whether the view is attached to another multi_view.
        """
        try:
            return self._attached_primal_param is not None
        except AttributeError:
            return False

    def attach_to(self, weight_fn) -> None:
        self._logger.debug(self.__repr__() + " is attached to a multi-view.")
        assert not self.attached, 'Cannot attach a view which is already attached'
        self._attached_primal_param = weight_fn

    def detach(self) -> None:
        self._logger.debug(self.__repr__() + ' is now detached.')
        w = self.primal_param
        self._attached = False
        self.primal_param = w

    ## WEIGHT
    @property
    def primal_param(self) -> Tensor:
        r"""
        Primal parameters of size [dim_feature, dim_output].
        """
        if self.attached:
            return self._attached_primal_param()
        else:
            if self._primal_param_exists:
                return self._primal_param.T
            raise utils.NotInitializedError(cls=self, message="No weight has been initialized or computed yet.")

    @primal_param.setter
    def primal_param(self, val):
        if self.attached:
            raise Exception("Cannot assign value of an attached view. Please change the value of the mother class "
                            "instead.")
        else:
            self._reset_dual()
            if val is not None:
                with torch.no_grad():
                    val = utils.castf(val, tensor=False, dev=self._primal_param.device)
                    if self._primal_param_exists and self._primal_param.shape == val.shape:
                        self._primal_param.copy_(val.T)
                        # zeroing the gradients if relevant
                        if self._level_trainable and self._primal_param.grad is not None:
                            self._primal_param.grad.data.zero_()
                    else:
                        del self._primal_param
                        # torch.no_grad() does not affect the constructor
                        self._primal_param = torch.nn.Parameter(val.T, requires_grad=self._level_trainable)

                    self._dim_output = self.primal_param.shape[1]
            else:
                self._reset_primal()
                self._logger.info("The weight is unset.")

    @property
    def _primal_param_exists(self) -> bool:
        try:
            return self._primal_param.nelement() != 0
        except AttributeError:
            return False

    def _update_primal_from_dual(self):
        raise NotImplementedError

    @abstractmethod
    def _update_dual_from_primal(self):
        raise NotImplementedError

    @property
    def _param_device(self) -> torch.device:
        return self._dual_param.device

    @abstractmethod
    def phi(self, x=None) -> Tensor:
        pass

    @abstractmethod
    def k(self, x=None) -> Tensor:
        pass

    def c(self, x=None) -> Tensor:
        phi = self.phi(x)
        return phi.T @ phi / self.num_idx

    def phiw(self, x=None, representation=None) -> Tensor:
        r"""
        :param x: Input, defaults to the sample (None).
        :type x: torch.Tensor [num, dim_input]
        :param representation: 'primal' or 'dual'.
        :type representation: str, optional
        :return: :math:`\phi(x)^\top W` or :math:`k(x)^top H`
        :rtype: torch.Tensor [num, dim_output]
        """
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
    @abstractmethod
    def H(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def W(self) -> Tensor:
        pass

    @property
    def PhiW(self) -> Tensor:
        return self.phiw()

    @property
    def dual_correlation(self) -> Tensor:
        r"""
        Correlation of the hidden variables :math:`\mathbf{h}^\top \mathbf{h}`. This should be the identity provided
        that the hidden variables lie on the Stiefel manifold.
        """
        return self.dual_param.T @ self.dual_param

    @property
    def primal_correlation(self) -> Tensor:
        r"""
        Correlation of the weights :math:`\mathbf{w}^\top \mathbf{w}`. This should be the identity provided
        that the weights lie on the Stiefel manifold.
        """
        return self.primal_param.T @ self.primal_param

    @property
    def dual_projector(self) -> Tensor:
        r"""
        Projector on the subspace spanned by the hidden variables :math:`\mathbf{h}\mathbf{h}^\top`.
        This is a rigorous projector provided its determinant is unity, e.g. when the hidden variables lie on the
        Stiefel manifold.
        """
        return self.dual_param @ self.dual_param.T

    @property
    def primal_projector(self) -> Tensor:
        r"""
        Projector on the subspace spanned by the weights :math:`\mathbf{w}\mathbf{w}^\top`.
        This is a rigorous projector provided its determinant is unity, e.g. when the weights lie on the Stiefel manifold.
        """
        return self.primal_param @ self.primal_param.T

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
