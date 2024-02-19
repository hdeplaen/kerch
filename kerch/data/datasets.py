# coding=utf-8
import os
import numpy as np
import torch

from ..utils import kwargs_decorator, FTYPE
from ._LearningSet import _LearningSetTrain, _LearningSetTrainTest

def _get_file_path(name: str):
    dir_path = os.path.dirname(__file__)
    return os.path.join(dir_path, 'files', name)


class TwoMoons(_LearningSetTrain):
    def __init__(self, *args, **kwargs):
        self._noise = kwargs.pop('noise', .1)
        self._separation = kwargs.pop('separation', [1, .5])
        assert isinstance(self._noise, (int, float))
        assert self._noise >= 0.
        assert isinstance(self._separation, (list, tuple))
        assert len(self._separation) == 2

        # range computation
        min_x = min(-1, self._separation[0]-1) - 2. * self._noise
        max_x = max(1, self._separation[0]+1) + 2. * self._noise
        min_y = min(0, self._separation[1]-1) - 2. * self._noise
        max_y = max(1, self._separation[1]) + 2. * self._noise

        fact_x = (max_x - min_x) / 2
        fact_y = (max_y - min_y) / 2

        range = (min_x - fact_x, max_x + fact_x,
                 min_y - fact_y, max_y + fact_y)

        super(TwoMoons, self).__init__(name="Two Moons",
                                       dim_data=2,
                                       dim_labels=1,
                                       range=range,
                                       **kwargs)


    def _training(self, num):
        n_samples_out = num // 2
        n_samples_in = num - n_samples_out

        outer_circ_x = torch.cos(torch.linspace(0, torch.pi, n_samples_out))
        outer_circ_y = torch.sin(torch.linspace(0, torch.pi, n_samples_out))
        inner_circ_x = self._separation[0] - torch.cos(torch.linspace(0, torch.pi, n_samples_in))
        inner_circ_y = self._separation[1] - torch.sin(torch.linspace(0, torch.pi, n_samples_in))

        data = torch.cat((torch.stack([inner_circ_x, inner_circ_y], dim=1),
                          torch.stack([outer_circ_x, outer_circ_y], dim=1)))
        labels = torch.cat((torch.ones(n_samples_in, dtype=FTYPE), torch.zeros(n_samples_out, dtype=FTYPE)))[:, None]

        if self._noise != 0.:
            data += self._noise * torch.randn(data.shape)

        return data, labels


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

