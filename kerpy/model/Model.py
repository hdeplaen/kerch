import numpy as np
import torch
from abc import ABCMeta, abstractmethod

from .._module import _module
from .. import utils

class Model(_module, metaclass=ABCMeta):
    @abstractmethod
    @utils.kwargs_decorator({
        "loss": torch.nn.MSELoss
    })
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self._loss = kwargs["loss"]

        self._training_data = None
        self._training_labels = None
        self._validation_data = None
        self._validation_labels = None
        self._testing_data = None
        self._testing_labels = None

    def loss(self, data, labels):
        pred = self.forward(data)
        return self._loss(pred, labels)

    def fit(self, training_data=None, training_labels=None):
        if training_data is None:
            training_data = self._training_data
            training_labels = self._training_labels
        self.solve(training_data, training_labels)

    def validate(self):

    def set_data_raw(self, training_data=None, training_labels=None,
                           validation_data=None, validation_labels=None,
                           testing_data=None, testing_labels=None):
        self._training_data = utils.castf(training_data)
        self._training_labels = utils.castf(training_labels)
        self._validation_data = utils.castf(validation_data)
        self._validation_labels = utils.castf(validation_labels)
        self._testing_data = utils.castf(testing_data)
        self._testing_labels = utils.castf(testing_labels)

    def create_val(self, prop_of_training=.2):

    def set_data_prop(self, data=None, labels=None, proportions=None):
        if proportions is None:
            proportions = [.7, .15, .15]

        data_list, labels_list = Model._split_data(data, labels, proportions)

        self.set_data_raw(training_data=data_list[0], training_labels=labels_list[0],
                          validation_data=data_list[1], validation_labels=labels_list[1],
                          testing_data=data_list[2], testing_labels=labels_list[2])

    def hyperopt(self, model_params, kernel_params):
        for key, value in model_params:
            if value is None:
                try:
                    base_value = self.__getattr__(key)
                    model_params[key] = base_value
                except AttributeError:
                    self._log.warning(f"Cannot optimize parameter {key} as it appears to not exist in this model. "
                                      f"Maybe this is a kernel parameter and not a model parameter.")

        for key, value in kernel_params:
            if value is None:
                try:
                    base_value = self.__getattr__(key)
                    kernel_params[key] = base_value
                except AttributeError:
                    self._log.warning(f"Cannot optimize parameter {key} as it appears to not exist in the kernel "
                                      f"of this model.")

    ####################################################################################################################

    @staticmethod
    def _split_data(data, labels=None, props=None):
        r"""
        Splits the data in multiple random datasets based on props.
        """
        if props is None:
            return data, labels

        n = data.shape[0]
        perm = np.random.permutation(n)
        data = data[perm]
        if labels is not None:
            labels = labels[perm]

        data_list = []
        labels_list = []
        pos_start = 0
        for prop in props:
            pos_end = pos_start + round(prop * n)
            data_list.append(data[pos_start:pos_end])
            if labels is None:
                labels_list.append(labels[pos_start:pos_end])
            else:
                labels_list.append(None)

        return data_list, labels_list




