import torch
import logging
from abc import ABCMeta, abstractmethod


from .._module import _module
from .._dataholder import _dataholder
from .. import utils

class Model(_dataholder, _module, metaclass=ABCMeta):
    @abstractmethod
    @utils.kwargs_decorator({
        "loss": torch.nn.MSELoss(),
        "log_level": logging.ERROR
    })
    def __init__(self, **kwargs):
        _module.__init__(self, **kwargs)
        _dataholder.__init__(self, **kwargs)
        self._loss = kwargs["loss"]
        self.set_log_level()

    @abstractmethod
    def fit(self, data=None, labels=None) -> None:
        pass

    def error(self, data=None, labels=None):
        data, labels = self._get_default_data(data, labels)
        pred = self.forward(data)
        return self._loss(pred, labels)

    def validate(self, k:int=0, prop:float=.2):
        if k==0:
            self.fit()
            return self.error(self._validation_data, self._validation_labels)
        else:
            loss = 0.
            if self._validation_data is not None:
                self.info("The validation set is not used for k-fold cross-validation, "
                          "only the training set is divided.")
            for fold in range(k):
                data_list, labels_list = _dataholder._get_fold(prop=prop)
                self.fit(data_list[0], labels_list[0])
                loss += self.error(data_list[1], labels_list[1])
            return loss/k

    def hyperopt(self, params, k:int=0, max_evals=1000):
        r"""
        Optimizes the hyperparameters of the model based on a random grid search.
        """

        # we only import the packages in hyperopt
        from hyperopt import hp, fmin, tpe, STATUS_OK
        from math import log
        import copy

        _KERNEL_IDENTIFIER = "__KERNEL__ "
        _PARAM_LOG_RANGE = 3

        ## OBJECTIVE FUNCTION
        def objective(*args,**kwargs):
            model = copy.deepcopy(self)
            self._log.debug("Executing a hyperparameter trial on a copy of the model.")
            for key, value in kwargs:
                if key.startswith(_KERNEL_IDENTIFIER):
                    model.kernel.__setattr__(key.split()[1], value)
                else:
                    model.__setattr__(key, value)
            return {'loss': model.validate(k=k),
                    'status': STATUS_OK}

        ## PARAMETER SPACE
        def _add_range(key, base_value):
            if type(base_value)==bool:
                return hp.choice([True, False])
            else:
                return hp.loguniform(key, base_value - _PARAM_LOG_RANGE,
                                    base_value + _PARAM_LOG_RANGE)


        hp_params = {}
        for key in params:
            try: # first see if this is a parameter of the model
                base_value = log(getattr(self, key))
                hp_params.update({key: _add_range(key, base_value)})
            except AttributeError:
                try: # if not, this may be a parameter of the kernel
                    base_value = log(getattr(self.kernel, key))
                    kernel_key = _KERNEL_IDENTIFIER + key
                    hp_params.update({kernel_key: _add_range(kernel_key, base_value)})
                except AttributeError:
                    self._log.warning(f"Cannot optimize parameter {key} as it appears to not exist in this "
                                      f"model, neither in its kernel component.")

        if k==0:
            self._log.info("Using a classical validation scheme.")
        else:
            self._log.info(f"Using a {k}-fold cross-validation scheme.")

        ## OPERATE SEARCH
        # avoid printing everything multiple times during all the searches
        import logging
        _OLD_LEVEL = self._log.level
        _OLD_LEVEL_KERNEL = self.kernel._log.level
        self._log.setLevel(logging.ERROR)
        self.kernel._log.setLevel(logging.ERROR)

        best = fmin(objective, space=hp_params, algo=tpe.suggest, max_evals=max_evals)

        # reset the log levels
        self.set_log_level(_OLD_LEVEL)
        self.kernel.set_log_level(_OLD_LEVEL_KERNEL)

        # set the fond values
        for key, value in best.items():
            if key.startswith(_KERNEL_IDENTIFIER):
                kernel_key = key.split()[1]
                self.kernel.__setattr__(kernel_key, value)
                self._log.info(f"The new kernel value for {kernel_key} is {value}.")
            else:
                self.__setattr__(key, value)
                self._log.info(f"The new model value for {key} is {value}.")
