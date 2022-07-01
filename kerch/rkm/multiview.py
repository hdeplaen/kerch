"""
Abstract class for multi-View representations.
"""
from collections import OrderedDict

import torch
from torch import Tensor as T

from .. import utils
from .view import View
from .._stochastic import _stochastic


@utils.extend_docstring(_stochastic)
class MultiView(_stochastic):
    r"""
    TODO
    """

    @utils.kwargs_decorator({
        "dim_output": None,
        "hidden": None,
        "weight": None,
        "param_trainable": False,
        "prop": None
    })
    def __init__(self, *views, **kwargs):
        super(MultiView, self).__init__(**kwargs)
        self._dim_output = kwargs["dim_output"]
        self._views = OrderedDict()
        self._num_views = 0

        self._log.debug("The output dimension, the sample and the hidden variables of each View will be overwritten "
                        "by the general value passed as an argument, possibly with None.")

        # HIDDEN
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

        # append the views
        for view in views:
            self._add_view(view)

        self.stochastic(prop=kwargs["prop"])

    def __repr__(self):
        repr = ""
        for key, val in self.views.items():
            repr += "\n\t* " + key + ": " + val.__repr__()
        return repr

    def _add_view(self, view) -> None:
        """
        Adds a view
        """
        if isinstance(view, View):
            view.dim_output = self._dim_output
            view.hidden = self.hidden_as_param
        elif isinstance(view, dict):
            view = View(**{**view,
                           "dim_output": self._dim_output,
                           "hidden": self.hidden_as_param})
        else:
            self._log.error(f"View {view} could not be added as it is nor a view object nor a dictionnary of "
                            f"parameters")
            return

        if view.num_sample is not None and self._num_total is not None:
            assert view.num_sample == self._num_total, 'Inconsistency in the sample sizes of the initialized views.'
        elif view.num_sample is not None and self._num_total is None:
            self._num_total = view.num_sample

        # append to dict and meta variables for the views
        try:
            self._views[view.name] = view
        except AttributeError:
            name = str(self._num_views)
            self._views[name] = view
        self._num_views += 1

    def _reset_hidden(self) -> None:
        for view in self._views:
            view.hidden = self._reset_hidden()

    def _reset_weight(self) -> None:
        for view in self._views:
            view._reset_weight()

    @property
    def dims_feature(self) -> list:
        dims = []
        for num in range(self._num_views):
            dims.append(self.view(num).kernel.dim_feature)
        return dims

    @property
    def dim_feature(self) -> int:
        return sum(self.dims_feature)

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

    ## VIEWS
    def view(self, id) -> View:
        if isinstance(id, int):
            return list(self._views.items())[id][1]
        elif isinstance(id, str):
            return self._views[id]

    @property
    def views(self) -> OrderedDict:
        return self._views

    @property
    def num_views(self) -> int:
        return self._num_views

    ## HIDDEN

    @property
    def hidden(self) -> T:
        if self._hidden_exists:
            return self._hidden.data[self.idx, :]

    def update_hidden(self, val: T, idx_sample=None) -> None:
        # first verify the existence of the hidden values before updating them.
        if not self._hidden_exists:
            self._log.warning("Could not update hidden values as these do not exist. "
                              "Please set the values for hidden first.")
            return

        if idx_sample is None:
            idx_sample = self._all_sample()
        self._hidden.data[idx_sample, :] = val.data

        for view in self._views:
            view.hidden = self.hidden_as_param

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

            for _, view in self._views.items():
                view.hidden = self.hidden_as_param
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
    def weight(self) -> T:
        return self.weight_as_param.data

    @property
    def weight_as_param(self) -> torch.nn.Parameter:
        weight = self.view(0).weight_as_param
        for num in range(1, self.num_views):
            weight = torch.cat((weight, self.view(num).weight_as_param), dim=0)
        return weight

    def weight_from_views(self, views: list) -> torch.nn.Parameter:
        views = list(views)
        assert len(views) > 0, 'The number of views must be at least equal to one.'
        weight = self.view(views[0]).weight_as_param
        for num in range(1, len(views)):
            weight = torch.cat((weight, self.view(views[num]).weight_as_param), dim=0)
        return weight

    @weight.setter
    def weight(self, val):
        if val is not None:
            assert val.shape[0] == self.dim_feature, f'The feature dimensions do not match: ' \
                                                     f'got {val.shape[0]}, need {self.dim_input}.'
            previous_dim = 0
            for num in range(self.num_views):
                dim = self.dims_feature[num]
                self.view(num).weight = val[previous_dim:previous_dim + dim, :]
                previous_dim += dim
        else:
            self._log.info("The weight is unset.")
            for num in range(self.num_views):
                self.view(num).weight = None

    @property
    def _weight_exists(self) -> bool:
        return self._weight.nelement() != 0

    def _update_weight_from_hidden(self):
        for view in self._views:
            view._update_weight_from_hidden()

    ## MATHS
    def k(self, x=None) -> T:
        if not isinstance(x, dict):
            k = self.view(0).k(x)
            for num in range(1, self._num_views):
                k += self.view(num).k(x)
        else:
            items = list(x.items())
            key, value = items[0]
            k = self.view(key).k(value)
            for num in range(1, len(items)):
                key, value = items[num]
                k += self.view(key).k(value)
        return k

    def phi(self, x=None, features=None):
        # this returns classical phi
        if not isinstance(x, dict):
            phi = self.view(0).phi(x)
            for num in range(1, self._num_views):
                phi = torch.cat((phi, self.view(num).phi(x)), dim=1)
        # if different x values have to be used for specific views
        else:
            if features is None:
                features = []
            items = list(x.items())
            key, value = items[0]
            phi = self.view(key).phi(value)
            for num in range(1, len(items)):
                key, value = items[num]
                phi = torch.cat((phi, self.view(key).phi(value)), dim=1)
            for feat in features:
                phi = torch.cat((phi, feat), dim=1)
        return phi

    def c(self, x=None, features: list = None) -> T:
        phi = self.phi(x, features=features)
        return phi.T @ phi

    def h(self, x=None) -> T:
        if x is None:
            if self._hidden_exists:
                return self._hidden[self.idx, :]
            else:
                self._log.warning("No hidden values exist or have been initialized.")
                raise utils.DualError(self)
        raise NotImplementedError

    @property
    def K(self) -> T:
        k = self.view(0).K
        for num in range(1, self._num_views):
            k += self.view(num).K
        return k

    @property
    def C(self) -> T:
        # TODO: can probably be optimized
        return self.c()

    @property
    def H(self) -> T:
        return self.h()

    def forward(self, x=None, representation="dual"):
        raise NotImplementedError
