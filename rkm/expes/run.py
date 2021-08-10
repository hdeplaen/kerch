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
import numpy as np
from sklearn import preprocessing

from rkm.expes.data import data
import rkm.model.rkm as rkm


def lssvm():
    params = load_params('tests', 'lssvm')
    data_params = params['data']
    input, target, range = data.factory(data_params["dataset"],
                                        data_params["n_samples"])

    # SOFT RKM
    level_params_soft = params["level"]
    level_params_soft["constraint"] = "soft"
    mdl_soft = rkm.RKM(cuda=params["cuda"])
    mdl_soft.append_level(**level_params_soft)
    mdl_soft.learn(input, target, **params["opt"])

    # HARD RKM
    level_params_hard = params["level"]
    level_params_hard["constraint"] = "hard"
    mdl_hard = rkm.RKM(cuda=params["cuda"])
    mdl_hard.append_level(**level_params_hard)
    mdl_hard.learn(input, target, **params["opt"])

    print('LS-SVM test finished')


def kpca():
    params = load_params('tests', 'kpca')
    data_params = params['data']
    input, target, range = data.factory(data_params["dataset"],
                                        data_params["n_samples"])

    # SOFT RKM
    level_params_soft = params["level"]
    level_params_soft["constraint"] = "soft"
    mdl_soft = rkm.RKM(cuda=params["cuda"])
    mdl_soft.append_level(**level_params_soft)
    mdl_soft.learn(input, target, **params["opt"])
    print('Soft KPCA tested')

    # HARD RKM
    level_params_hard = params["level"]
    level_params_hard["constraint"] = "hard"
    mdl_hard = rkm.RKM(cuda=params["cuda"])
    mdl_hard.append_level(**level_params_hard)
    mdl_hard.learn(input, target, **params["opt"])
    print('Hard KPCA finished')

def pima_indians():
    # PRELIMINARIES
    params = load_params('multi', 'pima_indians')
    data_params = params['data']

    def _single_pima(num=0):
        print(f"ITERATION NUMBER {num}")

        training, validation, test, _ = data.factory(data_params["dataset"],
                                                     data_params["training"],
                                                     data_params["validating"],
                                                     data_params["testing"])

        # PREPROCESSING
        tr_input, tr_target = training
        val_input, val_target = validation
        test_input, test_target = test

        scaler = preprocessing.StandardScaler().fit(tr_input)
        tr_input = scaler.transform(tr_input)
        val_input = scaler.transform(val_input)
        test_input = scaler.transform(test_input)

        # SETTING MODEL UP
        mdl = rkm.RKM(cuda=params["cuda"])
        mdl.append_level(**params["level1"])
        mdl.append_level(**params["level2"])
        mdl.append_level(**params["level3"])
        print(mdl)

        # TRAINING
        mdl.learn(tr_input, tr_target, verbose=False,
                  val_x=val_input, val_y=val_target,
                  test_x=test_input, test_y=test_target,
                  **params["opt"])
        print(mdl)

        print("###########################################################")

    try:
        num_iter = params["num_iter"]
    except KeyError:
        num_iter = 1

    print(f"TOTAL NUMBER OF ITERATIONS: {num_iter}")
    for iter in range(0, num_iter):
        _single_pima(iter)

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

        training, validation, test, _ = data.factory(data_params["dataset"],
                                                     data_params["training"],
                                                     data_params["validating"],
                                                     data_params["testing"])

        # PREPROCESSING
        tr_input, tr_target = training
        val_input, val_target = validation
        test_input, test_target = test

        scaler = preprocessing.StandardScaler().fit(tr_input)
        tr_input = scaler.transform(tr_input)
        val_input = scaler.transform(val_input)
        test_input = scaler.transform(test_input)

        # SETTING MODEL UP
        mdl = rkm.RKM(cuda=params["cuda"])

        level_num = 1
        while True:
            level_name = f"level{level_num}"
            try:
                level_params = params[level_name]
            except KeyError:
                break
            default_level_params = {"init_kernels": data_params["training"]}
            level_params = {**default_level_params, **level_params}
            mdl.append(**level_params)
            level_num += 1

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


#######################################################################################################################

def load_params(file: str, expe: str):
    """
    Loads the parameters from the expe.yaml file.
    :param expe: string representing parameters.
    :return: dictionnary of parameters.
    """
    with open(os.path.join(sys.path[0], "rkm/expes/expes/" + file + ".yaml"), "r") as file:
        content = yaml.safe_load(file)
    return content.get(expe, 'Experiment not recognized in yaml file.')
