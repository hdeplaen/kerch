"""
Various datasets for experiments

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import pandas as pd
import numpy as np
import math
from sklearn import datasets, preprocessing


class data():
    @staticmethod
    def gaussians(tr_size=100, val_size=0, test_size=0):
        size = tr_size / 2
        s1, m1 = .7, (2, 1)
        s2, m2 = 1.2, (-2, -3)

        g1 = np.random.normal(m1, s1, (size, 2))
        g2 = np.random.normal(m2, s2, (size, 2))

        x = np.concatenate((g1[:, 0], g2[:, 0]))
        y = np.concatenate((g1[:, 1], g2[:, 1]))
        c = np.concatenate((np.repeat(0, size), np.repeat(1, size)))

        input = np.concatenate((g1, g2))
        target = np.concatenate((np.repeat(-1, size), np.repeat(1, size)))

        return [input, target], None, None, None

    @staticmethod
    def spiral(tr_size=194, val_size=0, test_size=0):
        size = tr_size / 2

        def spiral_xy(i, spiral_num):
            """
            Create the data for a spiral.

            Arguments:
                i runs from 0 to 96
                spiral_num is 1 or -1
            """
            φ = i / 16 * math.pi
            r = 70 * ((104 - i) / 104)
            x = (r * math.cos(φ) * spiral_num) / 13 + 0.5
            y = (r * math.sin(φ) * spiral_num) / 13 + 0.5
            return (x, y)

        def spiral(spiral_num):
            return [spiral_xy(i, spiral_num) for i in range(tr_size)]

        s1 = spiral(1)
        s2 = spiral(-1)

        input = np.concatenate((s1, s2))
        target = np.concatenate((np.repeat(-1, 97), np.repeat(1, 97)))
        r = (-5, 6, -5, 5)

        return [input, target], None, None, r

    @staticmethod
    def two_moons(tr_size=100, val_size=0, test_size=0):
        input, output = datasets.make_moons(tr_size, noise=.1)
        output = np.where(output == 0, -1, 1)
        range = (-4, 7, -4, 4)
        return [2.5 * input, output], None, None, range

    @staticmethod
    def usps(tr_size=100, val_size=0, test_size=0):
        digits = datasets.load_digits(2)
        x = digits['data']
        y = digits['target']
        y = np.where(y == 0, -1, 1)
        r = (0, 1, 0, 1)
        return [x[:tr_size, :, :], y[:tr_size]], None, None, r

    @staticmethod
    def pima_indians(tr_size=100, val_size=0, test_size=0):
        with open('rkm/expes/datasets/pima-indians-diabetes.csv') as csvfile:
            data = pd.read_csv(csvfile, delimiter=',',lineterminator='\n')
            # data = data.to_numpy()
        print('Pima Indians Diabetes dataset loaded. ')
        data = data.to_numpy()

        idx_random = np.random.permutation(767)
        idx_tr = idx_random[0:tr_size]
        idx_val = idx_random[tr_size:tr_size+val_size]
        idx_test = idx_random[tr_size+val_size:tr_size+val_size+test_size]

        training = [data[idx_tr,0:8], data[idx_tr,8]]
        validation = [data[idx_val, 0:8], data[idx_val, 8]]
        test = [data[idx_test, 0:8], data[idx_test, 8]]

        return training, validation, test, None

    @staticmethod
    def factory(name, tr_size=0, val_size=0, test_size=0):
        datasets = {"gaussians": data.gaussians,
                    "spiral": data.spiral,
                    "two_moons": data.two_moons,
                    "usps": data.usps,
                    "pima_indians": data.pima_indians}
        func = datasets.get(name, "Invalid dataset")
        if tr_size == 0:
            training, validation, test, range = func()
        else:
            training, validation, test, range = func(tr_size, val_size, test_size)

        return training, validation, test, range


