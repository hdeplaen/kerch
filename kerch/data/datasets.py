# coding=utf-8
import os
import numpy as np

from ..utils import kwargs_decorator
from ._LearningSet import _LearningSetTrain, _LearningSetTrainTest

def _get_file_path(name: str):
    dir_path = os.path.dirname(__file__)
    return os.path.join(dir_path, 'files', name)


class TwoMoons(_LearningSetTrain):
    @kwargs_decorator({'noise': .1})
    def __init__(self, *args, **kwargs):
        super(TwoMoons, self).__init__(name="Two Moons",
                                       dim_data=2,
                                       dim_labels=1,
                                       range=(-4, 7, -4, 4),
                                       **kwargs)
        self._noise = kwargs['noise']

    def _training(self, num):
        try:
            from sklearn import datasets as skdata
        except ModuleNotFoundError:
            raise ModuleNotFoundError("This dataset requires sklearn to be installed. Please install it and try again.")
        data, labels = skdata.make_moons(num, noise=self._noise)
        labels = np.where(labels == 0, -1, 1)
        return data * 2.5, labels


class PimaIndians(_LearningSetTrain):
    def __init__(self, *args, **kwargs):
        super(PimaIndians, self).__init__(name="Pima Indian Diabetes",
                                          dim_data=8,
                                          dim_labels=1,
                                          **kwargs)

    def _training(self, num):
        path = _get_file_path('pima.csv')
        diabetes = np.genfromtxt(path, delimiter=',')
        data = diabetes[:, :self._dim_data]
        labels = diabetes[:, self._dim_data:]
        labels = np.where(labels == 0, -1, 1)
        return data, labels


class Spiral(_LearningSetTrain):
    def __init__(self, *args, **kwargs):
        super(Spiral, self).__init__(name="Spiral",
                                          dim_data=2,
                                          dim_labels=1,
                                     range=(-5, 6, -5, 5),
                                     **kwargs)

    def _training(self, num):
        import math
        size = -(num // -2) # ceil integer division

        def spiral_xy(i, spiral_num):
            """
            Create the value for a spiral.

            Arguments:
                i runs from 0 to 96
                spiral_num is 1 or -1
            """
            phi = i / 16 * math.pi
            r = 70 * ((104 - i) / 104)
            x = (r * math.cos(phi) * spiral_num) / 13 + 0.5
            y = (r * math.sin(phi) * spiral_num) / 13 + 0.5
            return (x, y)

        def spiral(spiral_num):
            return [spiral_xy(i, spiral_num) for i in range(size)]

        s1 = spiral(1)
        s2 = spiral(-1)

        data = np.concatenate((s1, s2))
        labels = np.concatenate((np.repeat(-1, size), np.repeat(1, size)))
        return data, labels

class Gaussians(_LearningSetTrain):
    def __init__(self, *args, **kwargs):
        super(Gaussians, self).__init__(name="Gaussians",
                                        dim_data=2,
                                        dim_labels=1,
                                        **kwargs)

    def _training(self, num):
        size = -(num // -2) # ceil integer division
        s1, m1 = .7, (2, 1)
        s2, m2 = 1.2, (-2, -3)

        g1 = np.random.normal(m1, s1, (size, 2))
        g2 = np.random.normal(m2, s2, (size, 2))

        data = np.concatenate((g1, g2))
        labels = np.concatenate((np.repeat(-1, size), np.repeat(1, size)))
        return data, labels

class Iris(_LearningSetTrain):
    def __init__(self, *args, **kwargs):
        super(Iris, self).__init__(name="Fisher's Iris",
                                   dim_data=4,
                                   dim_labels=3,
                                   **kwargs)

    def _training(self, num):
        path = _get_file_path('iris.csv')
        diabetes = np.genfromtxt(path, delimiter=',')
        data = diabetes[:, :self._dim_data]
        labels = diabetes[:, self._dim_data:]
        return data, labels

