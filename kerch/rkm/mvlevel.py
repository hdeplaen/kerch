import torch
from torch import Tensor as T
from abc import ABCMeta, abstractmethod

from .multiview import MultiView
from kerch import utils


class MVLevel(MultiView, metaclass=ABCMeta):
    r"""
    :param eta: :math:`\eta`., defaults to 1.
    :param representation: Chosen representation, "primal" or "dual"., defaults to "dual".

    :type eta: double, optional
    :type representation: str, optional
    """

    @utils.extend_docstring(MultiView)
    @utils.kwargs_decorator({
        "eta": 1.,
        "representation": "dual"
    })
    def __init__(self, *args, **kwargs):
        super(MVLevel, self).__init__(*args, **kwargs)
        self.eta = kwargs["eta"]
        self._representation = utils.check_representation(kwargs["representation"], cls=self)

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

    def solve(self, representation=None) -> None:
        r"""
        Fits the model according to the input ``sample`` and output ``target``. Many models have both a primal and
        a dual formulation to be fitted.

        :param representation: Representation of the model (``"primal"`` or ``"dual"``)., defaults to ``"dual"``.
        :type representation: str, optional
        """

        self._log.debug("The fitting is always done on the full sample dataset, regardless of the stochastic state.")
        # set the sample to input (always works for the underlying kernel)

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

    def loss(self, representation=None) -> T:
        raise NotImplementedError
