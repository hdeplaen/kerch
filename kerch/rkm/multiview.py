"""
Abstract class for multi-View representations.
"""
from collections import OrderedDict

import torch
from torch import Tensor as T

from .. import utils
from ._view import _View
from .view import View
from .._stochastic import _Stochastic


@utils.extend_docstring(_Stochastic)
class MultiView(_View):
    r"""
    TODO
    """

    @utils.kwargs_decorator({
        "prop": None
    })
    def __init__(self, *views, **kwargs):
        super(MultiView, self).__init__(**kwargs)
        self._views = OrderedDict()
        self._num_views = 0

        self._log.debug("The output dimension, the sample and the hidden variables of each View will be overwritten "
                        "by the general value passed as an argument, possibly with None.")

        # append the views
        for view in views:
            self._add_view(view)

        self.stochastic(prop=kwargs["prop"])

    def __str__(self):
        repr = ""
        for key, val in self.views.items():
            repr += "\n\t* " + key + ": " + val.__repr__()
        return repr

    def _add_view(self, view) -> None:
        """
        Adds a view
        """
        # get or create the view
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

        # verify the consistency of the new view with the previous ones if relevant
        if view.num_sample is not None and self._num_total is not None:
            assert view.num_sample == self._num_total, 'Inconsistency in the sample sizes of the initialized views.'
        elif view.num_sample is not None and self._num_total is None:
            self._num_total = view.num_sample

        # append to dict and meta variables for the views
        try:
            name = view.name
        except AttributeError:
            name = str(self._num_views)
        self._views[name] = view
        self.add_module(name, view)
        self._num_views += 1

    def _reset_hidden(self) -> None:
        for view in self._views:
            view.hidden = self._reset_hidden()

    def _reset_weight(self) -> None:
        for view in self._views.values():
            view._reset_weight()

    ##################################################################"
    @property
    def dims_feature(self) -> list:
        dims = []
        for num in range(self._num_views):
            dims.append(self.view(num).kernel.dim_feature)
        return dims

    @property
    def dim_feature(self) -> int:
        return sum(self.dims_feature)

    ## VIEWS
    def view(self, id) -> View:
        try:
            if isinstance(id, int):
                return list(self._views.values())[id]
            elif isinstance(id, str):
                return self._views[id]
        except NameError:
            raise NameError('The requested view does not exist.')

    @property
    def views(self) -> OrderedDict:
        return self._views

    @property
    def num_views(self) -> int:
        return self._num_views

    ######################################################################
    def update_hidden(self, val: T, idx_sample=None) -> None:
        super(MultiView, self).update_hidden(val, idx_sample)
        for view in self._views:
            view.hidden = self.hidden_as_param

    ## WEIGHT
    @property
    def weight(self) -> T:
        return super(MultiView, self).weight

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
        try:
            return self.weight.nelement() != 0
        except:
            return False

    def _update_weight_from_hidden(self):
        for view in self._views:
            view._update_weight_from_hidden()

    ## MATHS
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

    def forward(self, x=None, representation="dual"):
        raise NotImplementedError