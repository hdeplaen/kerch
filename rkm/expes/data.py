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
    def gaussians(tot_data):
        size = tot_data / 2
        s1, m1 = .7, (2, 1)
        s2, m2 = 1.2, (-2, -3)

        g1 = np.random.normal(m1, s1, (size, 2))
        g2 = np.random.normal(m2, s2, (size, 2))

        x = np.concatenate((g1[:, 0], g2[:, 0]))
        y = np.concatenate((g1[:, 1], g2[:, 1]))
        c = np.concatenate((np.repeat(0, size), np.repeat(1, size)))

        input = np.concatenate((g1, g2))
        target = np.concatenate((np.repeat(-1, size), np.repeat(1, size)))

        info = {"range": None,
                "size": 2}
        return (input, target), info

    @staticmethod
    def spiral(tot_data):
        size = tot_data / 2

        def spiral_xy(i, spiral_num):
            """
            Create the data for a spiral.

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
            return [spiral_xy(i, spiral_num) for i in range(tot_data)]

        s1 = spiral(1)
        s2 = spiral(-1)

        input = np.concatenate((s1, s2))
        target = np.concatenate((np.repeat(-1, 97), np.repeat(1, 97)))

        range = (-5, 6, -5, 5)
        info = {"range": range,
                "size": 2}

        return (input, target), info

    @staticmethod
    def two_moons(tot_data):
        input, output = datasets.make_moons(tot_data, noise=.1)
        output = np.where(output == 0, -1, 1)

        range = (-4, 7, -4, 4)
        info = {"range": range,
                "size": 2}

        return (2.5 * input, output), info

    @staticmethod
    def usps(tot_data):
        digits = datasets.load_digits(2)
        x = digits['data'][:tot_data, :, :]
        y = digits['target'][:tot_data]
        y = np.where(y == 0, -1, 1)

        range = (0, 1, 0, 1)
        info = {"range": range,
                "size": x.shape[1]*x.shape[2]}
        return (x, y), info

    @staticmethod
    def pima_indians(tot_data):
        with open('rkm/expes/datasets/pima-indians-diabetes.csv') as csvfile:
            data = pd.read_csv(csvfile, delimiter=',', lineterminator='\n')
        print('Pima Indians Diabetes dataset loaded. ')
        data = data.to_numpy()

        input = data[0:tot_data, 0:8]
        target = data[0:tot_data, 8]

        info = {"range": None,
                "size": 8}
        return (input, target), info

    @staticmethod
    def generate_dataset(fun, tr_size=0, val_size=0, test_size=0, tot_data=None):
        # PREALLOC
        tr_input = []
        tr_target = []
        val_input = []
        val_target = []
        test_input = []
        test_target = []

        # LOAD
        if tot_data is None:
            tot_data = tr_size + val_size + test_size
        else:
            assert tot_data >= tr_size + val_size + test_size, \
                "Not enough data in the dataset for the requested sample sizes."

        assert tot_data is not 0, "Cannot select no data."
        dataset, info = fun(tot_data)
        input, target = dataset
        print("Data loaded.")

        # SELECT
        idx_random = np.random.permutation(tot_data)
        if tr_size is not 0:
            idx_tr = idx_random[0:tr_size]
            tr_input = input[idx_tr, :]
            tr_target = target[idx_tr, :]
        if val_size is not 0:
            idx_val = idx_random[tr_size:tr_size + val_size]
            val_input = input[idx_val, :]
            val_target = target[idx_val, :]
        if test_size is not 0:
            idx_test = idx_random[tr_size + val_size:tr_size + val_size + test_size]
            test_input = input[idx_test, :]
            test_target = target[idx_test, :]

        return (tr_input, tr_target), (val_input, val_target), (test_input, test_target), info

    @staticmethod
    def factory(name, tr_size=0, val_size=0, test_size=0):
        datasets = {"gaussians": (data.gaussians, None),
                    "spiral": (data.spiral, None),
                    "two_moons": (data.two_moons, None),
                    "usps": (data.usps, 5000),
                    "pima_indians": (data.pima_indians, 762)}
        fun, tot_data = datasets.get(name, "Invalid dataset")
        return data.generate_dataset(fun, tr_size, val_size, test_size, tot_data)
