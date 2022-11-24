import torch
from torch import Tensor as T
from abc import ABCMeta, abstractmethod

import kerch.opt
from ._view import _View
from kerch import utils


class _Level(_View, metaclass=ABCMeta):
    r"""
    :param eta: :math:`\eta`., defaults to 1.
    :param representation: Chosen representation, "primal" or "dual"., defaults to "dual".

    :type eta: double, optional
    :type representation: str, optional
    """

    @utils.extend_docstring(_View)
    @utils.kwargs_decorator({
        "eta": 1.,
        "representation": "dual"
    })
    def __init__(self, *args, **kwargs):
        super(_Level, self).__init__(*args, **kwargs)
        self.eta = kwargs["eta"]
        self._representation = utils.check_representation(kwargs["representation"], cls=self)

    @property
    def _I_primal(self) -> T:
        r"""
            To avoid multiple reinitializations, certainly when working on GPU, the value is stored in memory the first
            time it is called, to be re-used later."
        """
        if "I_primal" not in self._cache:
            self._cache["I_primal"] = torch.eye(self.dim_feature, dtype=utils.FTYPE)
        return self._cache["I_primal"]

    @property
    def _I_dual(self) -> T:
        r"""
                To avoid multiple reinitializations, certainly when working on GPU, the value is stored in memory the first
                time it is called, to be re-used later."
        """
        if "I_dual" not in self._cache:
            self._cache["I_dual"] = torch.eye(self.num_idx, dtype=utils.FTYPE)
        return self._cache["I_dual"]

    ####################################################################################################################

    @abstractmethod
    def _solve_primal(self) -> None:
        r"""
        Solves the dual formulation on the sample.
        """
        pass

    @abstractmethod
    def _solve_dual(self) -> None:
        r"""
        Solves the primal formulation on the sample.
        """
        pass

    def solve(self, sample=None, targets=None, representation=None, **kwargs) -> None:
        r"""
        Fits the model according to the input ``sample`` and output ``target``. Many models have both a primal and
        a dual formulation to be fitted.

        :param representation: Representation of the model (``"primal"`` or ``"dual"``)., defaults to ``"dual"``.
        :type representation: str, optional
        """

        # self._log.debug("The fitting is always done on the full sample dataset, regardless of the stochastic state.")
        # set the sample to input (always works for the underlying kernel)

        if sample is not None:
            self.init_sample(sample)
        if targets is not None:
            self.targets = targets

        # verify that the sample has been initialized
        if self._num_total is None:
            self._log.error("Cannot perform fitting as no input has been provided nor a sample already exists")
            return

        # check the representation is correct and set it to the default Level value if None
        representation = utils.check_representation(representation, default=self._representation, cls=self)

        # execute the corresponding fitting
        switcher = {"primal": self._solve_primal,
                    "dual": self._solve_dual}
        fun = switcher.get(representation)
        return fun()

    ####################################################################################################################

    @utils.kwargs_decorator({
        "method": "exact",
    })
    def fit(self, **kwargs):
        switcher = {"exact": self.solve,
                    "optimize": self.optimize}
        switcher.get(kwargs["method"], 'Invalid method type (must be solve or optimize')(**kwargs)

    @utils.kwargs_decorator({
        "representation": None,
        "loss": "classic",
        "maxiter": 10000,
        "verbose": True
    })
    def optimize(self, **kwargs) -> None:
        representation = utils.check_representation(kwargs["representation"], self._representation, cls=self)

        # PRELIMINARIES
        self.init_parameters(representation, overwrite=False)
        opt = kerch.opt.Optimizer(self, **kwargs)

        # TRAINING LOOP
        verbose = kwargs["verbose"]
        maxiter = kwargs["maxiter"]
        if verbose:
            from tqdm import trange
            bar = trange(maxiter)
        else:
            bar = range(maxiter)
        for epoch in bar:
            l = self.loss()
            opt.zero_grad()
            l.backward()
            opt.step()
            self.after_step()

            if epoch % 50 == 0 and verbose:
                bar.set_description(f"loss: {l}")

    ####################################################################################################################

    @abstractmethod
    def loss(self, representation=None) -> T:
        pass

    def after_step(self) -> None:
        r"""
            Perform after-step operations, for example a projection of the parameters onto some manifold.
        """
        pass
