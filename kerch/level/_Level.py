import torch
from torch import Tensor as T
from abc import ABCMeta, abstractmethod

from ._View import _View
from .. import utils, opt


class _Level(_View, metaclass=ABCMeta):
    r"""
    A level consists in a view (kernel + weight/hidden) with various optimization attributes.

    :param eta: :math:`\eta`., defaults to 1.
    :param representation: Chosen representation, "primal" or "dual"., defaults to "dual".

    :type eta: double, optional
    :type representation: str, optional
    """

    @utils.extend_docstring(_View)
    @utils.kwargs_decorator({
        "eta": 1.,
    })
    def __init__(self, *args, **kwargs):
        super(_Level, self).__init__(*args, **kwargs)
        self.eta = kwargs["eta"]
        self._parameter_related_cache = []

    @property
    def _I_primal(self) -> T:
        r"""
            To avoid multiple reinitializations, certainly when working on GPU, the value is stored in memory the first
            time it is called, to be re-used later."
        """
        return self._get("_I_primal", level_key="I_primal", fun=lambda: torch.eye(self.dim_feature, dtype=utils.FTYPE))

    @property
    def _I_dual(self) -> T:
        r"""
                To avoid multiple reinitializations, certainly when working on GPU, the value is stored in memory the first
                time it is called, to be re-used later."
        """
        return self._get("_I_dual", level_key="I_dual", fun=lambda: torch.eye(self.num_idx, dtype=utils.FTYPE))

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

    @torch.no_grad()
    def solve(self, sample=None, target=None, representation=None, **kwargs) -> None:
        r"""
        Fits the model according to the input ``sample`` and output ``target``. Many models have both a primal and
        a dual formulation to be fitted.

        :param representation: Representation of the model (``"primal"`` or ``"dual"``)., defaults to ``"dual"``.
        :type representation: str, optional
        """

        # self._log.debug("The fitting is always done on the full sample data, regardless of the stochastic state.")
        # set the sample to input (always works for the underlying kernel)

        if sample is not None:
            self.init_sample(sample)
        if target is not None:
            self.target = target

        # verify that the sample has been initialized
        if self._num_total is None:
            self._log.error("Cannot solve as no input has been provided nor a sample already exists")
            return

        # check the representation is correct and set it to the default Level value if None
        representation = utils.check_representation(representation, default=self._representation, cls=self)

        # execute the corresponding fitting
        switcher = {"primal": self._solve_primal,
                    "dual": self._solve_dual}

        switcher.get(representation)()

    ####################################################################################################################

    @abstractmethod
    def loss(self, representation=None) -> T:
        pass

    def after_step(self) -> None:
        r"""
            Perform after-step operations, for example a projection of the parameters onto some manifold.
        """
        self._reset_parameter_related_cache()

    def _reset_parameter_related_cache(self) -> None:
        self._remove_from_cache(self._parameter_related_cache)
