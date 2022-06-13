import torch
from torch import Tensor as T
from abc import ABCMeta, abstractmethod

from .view import view
from kerpy import utils


class level(view, metaclass=ABCMeta):
    r"""
    :param eta: :math:`\eta`., defaults to 1.
    :param representation: Chosen representation, "primal" or "dual"., defaults to "dual".

    :type eta: double, optional
    :type representation: str, optional
    """

    @utils.extend_docstring(view)
    @utils.kwargs_decorator({
        "eta": 1.,
        "representation": "dual"
    })
    def __init__(self, **kwargs):
        super(level, self).__init__(**kwargs)
        self.eta = kwargs["eta"]
        self._representation = utils.check_representation(kwargs["representation"], cls=self)

    ####################################################################################################################

    @abstractmethod
    def _solve_primal(self, target=None) -> None:
        r"""
        Solves the dual formulation on the sample.
        """
        pass

    @abstractmethod
    def _solve_dual(self, target=None) -> None:
        r"""
        Solves the primal formulation on the sample.
        """
        pass

    def solve(self, sample=None, target=None, representation=None) -> None:
        r"""
        Fits the model according to the input ``sample`` and output ``target``. Many models have both a primal and
        a dual formulation to be fitted.

        :param sample: Input sample of the model., defaults to the sample provided by the model.
        :param target: Target sample of the model, defaults to ```None``
        :param representation: Representation of the model (``"primal"`` or ``"dual"``)., defaults to ``"dual"``.

        :type sample: Matrix, optional
        :type target: Matrix or vector, optional
        :type representation: str, optional
        """

        self._log.debug("The fitting is always done on the full sample dataset, regardless of the stochastic state.")
        # set the sample to input (always works for the underlying kernel)
        if sample is not None:
            self._log.info("Setting the sample to the provided input. Possibly overwriting a previous one.")
            self.init_sample(sample) # keeping the stochastic state if set.

        # verify that the sample has been initialized
        try:
            sample = self.sample
        except AttributeError:
            self._log.error("Cannot perform fitting as no input has been provided nor a sample already exists")
            return

        # verify that the output has the same dimensions
        if target is not None:
            target = utils.castf(target)
            same_dim = sample.shape[0] == target.shape[0]
            if not same_dim:
                self._log.error("The number of sample points is not consistent with the output dimensions")
                return

        # check the representation is correct and set it to the default level value if None
        representation = utils.check_representation(representation, default=self._representation, cls=self)

        # execute the corresponding fitting
        switcher = {"primal": self._solve_primal,
                    "dual": self._solve_dual}
        fun = switcher.get(representation)
        return fun(target=target)

    ####################################################################################################################

    def loss(self, representation=None) -> T:
        raise NotImplementedError
