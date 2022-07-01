import torch
import logging
from abc import ABCMeta, abstractmethod

from .._module import _Module
from .._dataholder import _DataHolder
from .. import utils


class Model(_DataHolder, _Module, metaclass=ABCMeta):
    @abstractmethod
    @utils.kwargs_decorator({
        "loss": torch.nn.MSELoss(reduction='sum'),
        "log_level": logging.ERROR
    })
    def __init__(self, **kwargs):
        _Module.__init__(self, **kwargs)
        _DataHolder.__init__(self, **kwargs)
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
            self.fit(self._training_data, self._training_labels)
            return self.error(self._validation_data, self._validation_labels)
        else:
            loss = 0.
            if self._validation_data is not None:
                self._log.info("The validation set is not used for k-fold cross-validation, "
                          "only the training set is divided.")
            for fold in range(k):
                data_list, labels_list = self._get_fold(prop=prop)
                self.fit(data_list[0], labels_list[0])
                loss += self.error(data_list[1], labels_list[1])
            return loss/k

    def hyperopt(self, params, k:int=0, max_evals=1000, log_range=2):
        r"""
        Optimizes the hyperparameters of the model based on a random grid search.
        """

        # we only import the packages in hyperopt
        import hyperopt
        from math import log
        import copy

        _KERNEL_IDENTIFIER = "__KERNEL__ "

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
                    'status': hyperopt.STATUS_OK}

        ## PARAMETER SPACE
        def _add_range(key, base_value):
            if type(base_value)==bool:
                return hyperopt.hp.choice([True, False])
            else:
                base_value = log(base_value)
                return hyperopt.hp.loguniform(key, base_value - log_range,
                                    base_value + log_range)


        hp_params = {}
        for key in params:
            try: # first see if this is a parameter of the model
                base_value = getattr(self, key)
                hp_params.update({key: _add_range(key, base_value)})
            except AttributeError:
                try: # if not, this may be a parameter of the kernel
                    base_value = getattr(self.kernel, key)
                    kernel_key = _KERNEL_IDENTIFIER + key
                    hp_params.update({kernel_key: _add_range(kernel_key, base_value)})
                except AttributeError:
                    self._log.warning(f"Cannot optimize parameter {key} as it appears to not exist in this "
                                      f"model, neither in its kernel component if it has one.")

        if k==0:
            self._log.info("Using a classical validation scheme.")
        else:
            self._log.info(f"Using a {k}-fold cross-validation scheme.")

        ## OPERATE SEARCH
        # avoid printing everything multiple times during all the searches
        import logging
        _OLD_LEVEL = self._log.level
        self._log.setLevel(logging.ERROR)

        best = hyperopt.fmin(objective,
                             space=hp_params,
                             algo=hyperopt.rand.suggest,
                             max_evals=max_evals)

        # reset the log levels
        self.set_log_level(_OLD_LEVEL)

        # set the fond values
        for key, value in best.items():
            if key.startswith(_KERNEL_IDENTIFIER):
                kernel_key = key.split()[1]
                self.kernel.__setattr__(kernel_key, value)
                self._log.warning(f"Kernel value for {kernel_key} is set to optimal value {value}.")
            else:
                self.__setattr__(key, value)
                self._log.warning(f"Model value for {key} is set to optimal value {value}.")
