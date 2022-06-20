"""
Abstract class for multi-View representations.
"""
from collections import OrderedDict

import torch
from torch import Tensor as T

from kerch import utils
from kerch.rkm import View
from kerch._stochastic import _stochastic
from kerch.utils import MultiViewError


@utils.extend_docstring(_stochastic)
class MultiView(_stochastic):
    r"""
    TODO
    """

    def __new__(cls, *args, **kwargs):
        if len(args)==0:
            return View(**kwargs)
        elif len(args)==1:
            if isinstance(args[0], View):
                return args[0]
            else:
                return View(**args[0])
        else:
            return super(MultiView, cls).__new__(cls)

    @utils.kwargs_decorator({
        "dim_output": None,
        "hidden": None,
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
        hidden = kwargs["hidden"]
        self._hidden = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE), self._param_trainable)
        if hidden is not None:
            self.hidden = hidden

        # append the views
        for view in views:
            self._add_view(view)

        self.stochastic(prop=kwargs["prop"])

    def _add_view(self, view) -> None:
        """
        Adds a view
        """
        if isinstance(view, View):
            view.dim_output = None
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

        # append to dict
        try:
            self._views[view.name] = view
        except AttributeError:
            name = str(len(self._views))
            self._views[name] = view

        self._num_views += 1

    def _reset_hidden(self) -> None:
        for view in self._views:
            view.hidden = self._reset_hidden()

    def _reset_weight(self) -> None:
        for view in self._views:
            view._reset_weight()

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

            for view in self._views:
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
    def _update_weight_from_hidden(self):
        for view in self._views:
            view._update_weight_from_hidden()

    ## MATHS

    def k(self, x=None) -> T:
        k = self.view(0).k(x)
        for num in range(1, self._num_views):
            k += self.view(num).k(x)
        return k

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
    def H(self) -> T:
        return self.h()

    def forward(self, x=None, representation="dual"):
        raise NotImplementedError