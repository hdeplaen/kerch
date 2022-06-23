from ..rkm import LSSVM
from .Model import Model


class LSSVM(LSSVM, Model):
    def __init__(self, **kwargs):
        super(LSSVM, self).__init__(**kwargs)
        # Model.__init__(self, **kwargs)
        self.kernel.set_log_level()

    def fit(self, data=None, labels=None) -> None:
        data, labels = self._get_default_data(data, labels)
        self.solve(data, labels)

    def hyperopt(self, params, k: int = 0, max_evals: int = 1000):
        if self._training_data is None:
            self._log.warning("Cannot optimize a model hyperparameters if no training dataset (and optionally "
                              "validation dataset) has been set.")
        self.init_sample(self._training_data)
        return super(LSSVM, self).hyperopt(params, k, max_evals)

