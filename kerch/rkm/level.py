from torch import Tensor as T
from abc import ABCMeta, abstractmethod

from ._level import _Level
from .view import View
from kerch import utils


class Level(_Level, View, metaclass=ABCMeta):

    @utils.extend_docstring(_Level)
    @utils.extend_docstring(View)
    def __init__(self, *args, **kwargs):
        super(Level, self).__init__(*args, **kwargs)

    ####################################################################################################################

    def solve(self, sample=None, target=None, representation=None, **kwargs) -> None:
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

        # set the sample to input (always works for the underlying kernel)
        if sample is not None:
            self._log.info("Setting the sample to the provided input. Possibly overwriting a previous one.")
            self.init_sample(sample)  # keeping the stochastic state if set.

        # verify that the output has the same dimensions
        if target is not None:
            target = utils.castf(target, tensor=True)
            same_dim = sample.shape[0] == target.shape[0]
            if not same_dim:
                self._log.error("The number of sample points is not consistent with the output dimensions")
                return

        # solve model
        return super(Level, self).solve(sample=sample,
                                        target=target,
                                        representation=representation,
                                        **kwargs)

    ####################################################################################################################

    @abstractmethod
    def loss(self, representation=None) -> T:
        pass
