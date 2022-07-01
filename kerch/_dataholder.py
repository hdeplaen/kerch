import torch
from torch import Tensor as T
import numpy as np

from . import utils
from ._logger import _Logger


class _DataHolder(_Logger):

    def __init__(self, **kwargs):
        super(_DataHolder, self).__init__(**kwargs)
        self._training_data = None
        self._training_labels = None
        self._validation_data = None
        self._validation_labels = None
        self._testing_data = None
        self._testing_labels = None

    def set_data(self, training_data=None, training_labels=None,
                 validation_data=None, validation_labels=None,
                 testing_data=None, testing_labels=None):
        r"""
        Sets the different datasets explicitly.
        """
        self._training_data = utils.castf(training_data)
        self._training_labels = utils.castf(training_labels)
        self._validation_data = utils.castf(validation_data)
        self._validation_labels = utils.castf(validation_labels)
        self._testing_data = utils.castf(testing_data)
        self._testing_labels = utils.castf(testing_labels)

        # not necessary:
        # if isinstance(self, _Sample):
        #     self.init_sample(self._training_data)

    def set_data_prop(self, data=None, labels=None, proportions=None) -> None:
        r"""
        Sets the different datasets based on proportions.
        """
        if proportions is None:
            proportions = [.7, .15, .15]
        else:
            assert len(proportions)==3, 'The proportions should contain 3 elements (training, validation and test). ' \
                                        'Please fill 0 if you do not want the create on of these sets'

        data_list, labels_list = _DataHolder._split_data(data, labels, proportions)

        self.set_data(training_data=data_list[0], training_labels=labels_list[0],
                      validation_data=data_list[1], validation_labels=labels_list[1],
                      testing_data=data_list[2], testing_labels=labels_list[2])

    def set_data_training_test(self, training_data=None, training_labels=None,
                                     testing_data=None, testing_labels=None,
                                     validation_prop=0.) -> None:
        r"""
        Sets the different datasets based on a training and a test set only. A validation set can be created from
        the training set based on some proportion.
        """
        data_list, labels_list = _DataHolder._split_data_two(training_data, training_labels, prop=validation_prop)
        training_data, validation_data = data_list
        training_labels, validation_labels = labels_list

        self.set_data(training_data, training_labels,
                      validation_data, validation_labels,
                      testing_data, testing_labels)

    def _get_fold(self, prop:float=.2):
        return _DataHolder._split_data_two(self._training_data, self._training_labels, prop=prop)

    def _get_default_data(self, data=None, labels=None):
        if data is None:
            data = self._training_data
            labels = self._training_labels
        elif labels is None: # if labels is None and not data
            raise NotImplementedError # reconstruction error for example in KPCA (still see how to implement it neatly).
        return data, labels

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
            if prop==0.:
                data_list.append(None)
                labels_list.append(None)
            else:
                pos_end = pos_start + round(prop * n)
                data_list.append(data[pos_start:pos_end])
                if labels is not None:
                    labels_list.append(labels[pos_start:pos_end])
                else:
                    labels_list.append(None)

        return data_list, labels_list

    @staticmethod
    def _split_data_two(data, labels=None, prop:float=.2):
        return _DataHolder._split_data(data, labels, props=[1. - prop, prop])
