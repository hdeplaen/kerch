# coding=utf-8
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from random import shuffle
from torch.utils.data import TensorDataset

from ..utils import kwargs_decorator, castf


class _LearningSet(metaclass=ABCMeta):
    @kwargs_decorator({'shuffle': True})
    def __init__(self, name: str, dim_data: int, dim_labels: int, range:int = None, **kwargs):
        super(_LearningSet, self).__init__()
        self._name = name
        self._dim_data = dim_data
        self._dim_labels = dim_labels
        self._plot_range: list[int] | None = range
        self._shuffle: bool = kwargs.pop('shuffle', True)

    def __str__(self):
        return self._name + \
            f" (Training: {len(self.training_set)}, Validation: {len(self.validation_set)}, Test: {len(self.test_set)})"

    def _init_datasets(self,
                       training_data, training_labels,
                       validation_data, validation_labels,
                       test_data, test_labels):
        assert training_data.shape[0] == training_labels.shape[0], \
            'The number of training datapoints and labels is inconsistent.'
        self._training_set = TensorDataset(castf(training_data), castf(training_labels))

        assert validation_data.shape[0] == validation_labels.shape[0], \
            'The number of validation datapoints and labels is inconsistent.'
        self._validation_set = TensorDataset(castf(validation_data), castf(validation_labels))

        assert test_data.shape[0] == test_labels.shape[0], \
            'The number of test datapoints and labels is inconsistent.'
        self._test_set = TensorDataset(castf(test_data), castf(test_labels))

    @property
    def info(self) -> dict:
        return {'name': self._name,
                'range': self._plot_range,
                'dim_data': self._dim_data,
                'dim_labels': self._dim_labels}

    @property
    def training_set(self) -> TensorDataset | None:
        return self._training_set

    @property
    def validation_set(self) -> TensorDataset | None:
        return self._validation_set

    @property
    def test_set(self) -> TensorDataset | None:
        return self._test_set

    @property
    def validation_exists(self) -> bool:
        return len(self._validation_set) > 0

    @property
    def test_exists(self) -> bool:
        return len(self._test_set) > 0

    def _indices(self, sizes: list[int], total=None) -> list[list]:
        r"""
        Splits the value in multiple random datasets based on props.
        """
        if total is None:
            total = sum(sizes)
        idx = list(range(total))
        if self._shuffle:
            shuffle(idx)

        idx_list = []
        pos_start = 0
        for size in sizes:
            if size == 0.:
                idx_list.append([])
            else:
                pos_end = pos_start + size
                idx_list.append(idx[pos_start:pos_end])
                pos_start = pos_end
        return idx_list


class _LearningSetTrain(_LearningSet, metaclass=ABCMeta):
    @kwargs_decorator({'num_training': 100,
                       'num_validation': 20,
                       'num_test': 20})
    def __init__(self, *args, **kwargs):
        super(_LearningSetTrain, self).__init__(*args, **kwargs)
        sizes = [kwargs['num_training'],
                 kwargs['num_validation'],
                 kwargs['num_test']]
        num = sum(sizes)

        data, labels = self._training(num)
        idx_list = self._indices(sizes, total=data.shape[0])
        idx_training = idx_list[0]
        idx_validation = idx_list[1]
        idx_test = idx_list[2]

        self._init_datasets(training_data=data[idx_training],
                            training_labels=labels[idx_training],
                            validation_data=data[idx_validation],
                            validation_labels=labels[idx_validation],
                            test_data=data[idx_test],
                            test_labels=labels[idx_test])

    @abstractmethod
    def _training(self, num):
        pass


class _LearningSetTrainTest(_LearningSet, metaclass=ABCMeta):
    @kwargs_decorator({'num_training': 100,
                       'num_validation': 20,
                       'num_test': 20})
    def __init__(self, *args, **kwargs):
        super(_LearningSetTrainTest, self).__init__(*args, **kwargs)
        sizes = [kwargs['num_training'],
                 kwargs['num_validation']]
        num = sum(sizes)

        training_data, training_labels = self._training(num)
        test_data, test_labels = self._test(kwargs['num_test'])
        idx_list = self._indices(sizes, total=training_data.shape[0])
        idx_training = idx_list[0]
        idx_validation = idx_list[1]
        idx_test = self._indices(kwargs['num_test'], total=test_data.shape[0])

        self._init_datasets(training_data=training_data[idx_training],
                            training_labels=training_labels[idx_training],
                            validation_data=training_data[idx_validation],
                            validation_labels=training_labels[idx_validation],
                            test_data=test_data[idx_test],
                            test_labels=test_labels[idx_test])

    @abstractmethod
    def _training(self, num):
        pass

    @abstractmethod
    def _test(self, num):
        pass
