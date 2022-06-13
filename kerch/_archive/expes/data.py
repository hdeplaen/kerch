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
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class data():
    @staticmethod
    def gaussians(tot_data, test_data):
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
        return (input, target), (None, None), info

    @staticmethod
    def spiral(tot_data, test_data):
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
            return [spiral_xy(i, spiral_num) for i in range(size)]

        s1 = spiral(1)
        s2 = spiral(-1)

        input = np.concatenate((s1, s2))
        target = np.concatenate((np.repeat(-1, 97), np.repeat(1, 97)))

        range = (-5, 6, -5, 5)
        info = {"range": range,
                "size": 2}

        return (input, target), (None, None), info

    @staticmethod
    def two_moons(tot_data, test_data):
        input, output = datasets.make_moons(tot_data, noise=.1)
        output = np.where(output == 0, -1, 1)

        range = (-4, 7, -4, 4)
        info = {"range": range,
                "size": 2}

        return (2.5 * input, output), (None, None), info

    @staticmethod
    def usps(tot_data, test_data):
        digits = datasets.load_digits(2)
        x = digits['data'][:tot_data, :, :]
        y = digits['target'][:tot_data]
        y = np.where(y == 0, -1, 1)

        range = (0, 1, 0, 1)
        info = {"range": range,
                "size": x.shape[1]*x.shape[2]}
        return (x, y), (None, None), info

    @staticmethod
    def pima_indians(tot_data, test_data):
        with open('kerpy/expes/datasets/pima-indians-diabetes.csv') as csvfile:
            data = pd.read_csv(csvfile, header=None, delimiter=',', lineterminator='\n')
        print('Pima Indians Diabetes dataset loaded. ')
        data = data.to_numpy()

        input = data[0:tot_data, 0:8]
        target = data[0:tot_data, 8]
        target[target==0] = -1

        info = {"range": None,
                "size": 8}
        return (input, target), (None, None), info

    @staticmethod
    def bupa_liver_disorder(tot_data, test_data):
        with open('kerpy/expes/datasets/pima-indians-diabetes.csv') as csvfile:
            data = pd.read_csv(csvfile, header=None, delimiter=',', lineterminator='\n')
        print('Bupa liver disorder dataset loaded. ')
        data = data.to_numpy()

        input = data[0:tot_data, 0:6]
        target = data[0:tot_data, 6]
        target[target==0] = -1

        info = {"range": None,
                "size": 6}
        return (input, target), (None, None), info

    @staticmethod
    def adult(tot_data, test_data):
        with open('kerpy/expes/datasets/adult_tr.csv') as csvfile:
            data_tr = pd.read_csv(csvfile, header=None, delimiter=',', lineterminator='\n')
        with open('kerpy/expes/datasets/adult_te.csv') as csvfile:
            data_te = pd.read_csv(csvfile, header=None, delimiter=',', lineterminator='\n')

        tr_input = data_tr.iloc[:, 0:13]
        tr_target = data_tr.iloc[:, -1].to_frame()
        te_input = data_te.iloc[:, 0:13]
        te_target = data_te.iloc[:, -1].to_frame()

        cat_idx_input = tr_input.select_dtypes(include=['object', 'bool']).columns
        steps_input = [('c', OneHotEncoder(handle_unknown='ignore'), cat_idx_input)]
        ct_input = ColumnTransformer(steps_input, sparse_threshold=0)
        tr_input = ct_input.fit_transform(tr_input)
        te_input = ct_input.fit_transform(te_input)

        tr_target = (tr_target == " <=50K").astype(int).values.flatten()
        te_target = (te_target == " <=50K").astype(int).values.flatten()

        print('Adult dataset loaded. ')

        info = {"range": None,
                "size": 60}
        return (tr_input, tr_target), (te_input, te_target), info

    @staticmethod
    def ionoshpere(tot_data, test_data):
        with open('kerpy/expes/datasets/ion.csv') as csvfile:
            data_tr = pd.read_csv(csvfile, header=None, delimiter=',', lineterminator='\n')

        tr_input = data_tr.iloc[:, 0:34].to_numpy()
        tr_target = data_tr.iloc[:, -1]
        tr_target = (tr_target == "g").astype(int).values.flatten()

        print('Ionosphere dataset loaded. ')

        info = {"range": None,
                "size": 34}
        return (tr_input, tr_target), (None, None), info

    @staticmethod
    def generate_dataset(fun, tr_size, val_size, test_size, tot_data, test_data):
        # PREALLOC
        tr_input = []
        tr_target = []
        val_input = []
        val_target = []
        test_input = []
        test_target = []

        # LOAD
        if test_data is None:
            if tot_data is None:
                tot_data = tr_size + val_size + test_size
            else:
                assert tot_data >= tr_size + val_size + test_size, \
                    "Not enough data in the dataset for the requested sample sizes."
        else:
            if tot_data is None:
                tot_data = tr_size + val_size
            else:
                assert tot_data >= tr_size + val_size, \
                    "Not enough data in the training dataset for the requested training and validation sizes."
                assert test_data >= test_size, \
                    "Not enough data in the test set for the requested test size."

        assert tot_data is not 0, "Cannot select no data."
        dataset, test, info = fun(tot_data, test_data)
        input, target = dataset
        test_input, test_target = test

        # print("Data loaded.")

        # SELECT
        idx_random = np.random.permutation(tot_data)
        if tr_size is not 0:
            idx_tr = idx_random[0:tr_size]
            tr_input = input[idx_tr, :]
            tr_target = target[idx_tr]
        if val_size is not 0:
            idx_val = idx_random[tr_size:tr_size + val_size]
            val_input = input[idx_val, :]
            val_target = target[idx_val]

        if test_data is not 0:
            if test_data is None:
                idx_test = idx_random[tr_size + val_size:tr_size + val_size + test_size]
            if test_data is not None:
                idx_test = np.random.permutation(test_data)
            test_input = input[idx_test, :]
            test_target = target[idx_test]

        return (tr_input, tr_target), (val_input, val_target), (test_input, test_target), info

    @staticmethod
    def factory(name, tr_size=0, val_size=0, test_size=0):
        datasets = {"gaussians": (data.gaussians, None, None),
                    "spiral": (data.spiral, None, None),
                    "two_moons": (data.two_moons, None, None),
                    "usps": (data.usps, 5000, None),
                    "pid": (data.pima_indians, 767, None),
                    "ion": (data.ionoshpere, 351, None),
                    "bld": (data.bupa_liver_disorder, 344, None),
                    "adult": (data.adult, 32559, 16279)}
        fun, tot_data, test_data = datasets.get(name, "Please provide the name of a valid dataset.")
        return data.generate_dataset(fun, tr_size, val_size, test_size, tot_data, test_data)
