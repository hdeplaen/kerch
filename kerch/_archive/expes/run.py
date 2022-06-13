"""
Various experiments

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import os
import sys
import yaml
import unittest
from sklearn import preprocessing

from kerch._archive.expes.tests import Suites
from kerch._archive.expes.data import data
from kerch._archive.model import rkm as rkm


def tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(Suites.levels_suite())

def general_expe(name, verbose=False):
    # DEFAULT PARAMETERS
    default_params = {"num_iter": 1}
    default_data_params = {"training": 100,
                           "validating": 100,
                           "testing": 100}

    # LOADING PARAMETERS
    params = {**default_params, **load_params('expe', name)}
    data_params = {**default_data_params, **params['data']}

    # SINGLE EXPERIMENT
    def _single_expe(num=0):
        print(f"ITERATION NUMBER {num+1}")

        training, validation, test, info = data.factory(data_params["dataset"],
                                                     data_params["training"],
                                                     data_params["validating"],
                                                     data_params["testing"])

        # PREPROCESSING
        tr_input, tr_target = training
        val_input, val_target = validation
        test_input, test_target = test

        scaler = preprocessing.StandardScaler().solve(tr_input)
        tr_input = scaler.transform(tr_input)
        val_input = scaler.transform(val_input)
        test_input = scaler.transform(test_input)

        # SETTING MODEL UP
        mdl = rkm.RKM(cuda=params["cuda"], name=name)

        level_num = 1
        size_params = {"size_in": info["size"],
                       "size_out": 1}
        while True:
            level_name = f"level{level_num}"
            try:
                level_params = params[level_name]
            except KeyError:
                break
            default_level_params = {"init_kernels": data_params["training"]}
            level_params = {**default_level_params, **size_params, **level_params}
            mdl.append_level(**level_params)
            level_num += 1
            size_params = {"size_in": level_params["size_out"]}

        print(mdl)

        # TRAINING
        mdl.learn(tr_input, tr_target, verbose=verbose,
                  val_x=val_input, val_y=val_target,
                  test_x=test_input, test_y=test_target,
                  **params["opt"])
        print(mdl)

        print("###########################################################")

    print(f"STARTING EXPERIMENT {name}")
    num_iter = params["num_iter"]
    print(f"TOTAL NUMBER OF ITERATIONS: {num_iter}")
    for iter in range(0, num_iter):
        _single_expe(iter)

def load_params(file: str, expe: str):
    """
    Loads the parameters from the expe.yaml file.
    :param expe: string representing parameters.
    :return: dictionnary of parameters.
    """
    with open(os.path.join(sys.path[0], "kerpy/expes/expes/" + file + ".yaml"), "r") as file:
        content = yaml.safe_load(file)
    return content.get(expe, 'Name of experiment not recognized in yaml file.')

def experiment(name):
    switcher = {"tests": tests}
    if name not in switcher:
        return general_expe(name)
    else:
        return switcher[name]()