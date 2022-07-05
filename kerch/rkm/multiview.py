"""
Abstract class for multi-View representations.
"""
from collections import OrderedDict
from typing import Iterator, List, Union, Tuple

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
        for key, val in self.named_views:
            repr += "\n\t " + key + ": " + val.__str__()
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
                           "hidden": self.hidden})
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
        super(MultiView, self)._reset_hidden()
        for v in self.views:
            v.hidden = self.hidden

    def _reset_weight(self) -> None:
        for view in self._views.values():
            view._reset_weight()

    ##################################################################"
    @property
    def dims_feature(self) -> List[int]:
        dims = []
        for v in self.views:
            dims.append(v.dim_feature)
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
    def views(self) -> Iterator[View]:
        for v in self._views.values():
            yield v

    @property
    def named_views(self) -> Iterator[Tuple[str, View]]:
        for n, v in self._views.items():
            yield n, v

    @property
    def num_views(self) -> int:
        return self._num_views

    ######################################################################
    ## WEIGHT
    @property
    def weights(self) -> Iterator[torch.nn.Parameter]:
        for v in self.views:
            yield v.weight

    @property
    def weight(self) -> T:
        return torch.cat(list(self.weights), dim=0)

    @weight.setter
    def weight(self, val):
        if val is not None:
            assert val.shape[0] == self.dim_feature, f'The feature dimensions do not match: ' \
                                                     f'got {val.shape[0]}, need {self.dim_input}.'
            previous_dim = 0
            for v in self.views:
                dim = v.dim_feature
                v.weight = val[previous_dim:previous_dim + dim, :]
                previous_dim += dim
        else:
            self._log.info("The weight is now unset.")
            for v in self.views:
                v.weight = None

    def _update_weight_from_hidden(self):
        for v in self._views:
            v._update_weight_from_hidden()

    def phis(self, x=None) -> Iterator[T]:
        if isinstance(x, T) or isinstance(x, torch.nn.Parameter) or x is None:
            for v in self.views:
                yield v.phi(x)
        elif isinstance(x, dict):
            for key, value in x.items():
                yield self.view(key).phi(value)
        else:
            raise NotImplementedError

    def phi(self, x=None) -> T:
        return torch.cat(list(self.phis(x)), dim=1)

    def ks(self, x=None) -> Iterator[T]:
        if isinstance(x, T) or isinstance(x, torch.nn.Parameter):
            for v in self.views:
                yield v.k(x)
        elif isinstance(x, dict):
            for key, value in x.items():
                yield self.view(key).k(value)
        else:
            raise NotImplementedError

    def k(self, x=None) -> T:
        return sum(self.ks(x))

    @property
    def Ks(self) -> T:
        for v in self.views:
            yield v.K

    @property
    def K(self) -> T:
        return sum(self.Ks)

    def forward(self, x=None, representation="dual"):
        raise NotImplementedError
