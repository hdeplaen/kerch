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
    mdl_soft.learn(input, target, maxiter=1e+4, tol=1e-7)

    # HARD RKM
    level_params_hard = params["level"]
    level_params_hard["constraint"] = "hard"
    mdl_hard = rkm.RKM(cuda=params["cuda"])
    mdl_hard.append_level(**level_params_hard)
    mdl_hard.learn(input, target, maxiter=1e+4, tol=1e-7)

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
    mdl_soft.learn(input, target, maxiter=1e+4, tol=1e-5)
    print('Soft KPCA tested')

    # HARD RKM
    level_params_hard = params["level"]
    level_params_hard["constraint"] = "hard"
    mdl_hard = rkm.RKM(cuda=params["cuda"])
    mdl_hard.append_level(**level_params_hard)
    mdl_hard.learn(input, target, maxiter=1e+2, tol=1e-4)
    print('Hard KPCA finished')

def pima_indians():
    params = load_params('two_levels', 'pima_indians')
    data_params = params['data']
    input, target, _ = data.factory(data_params["dataset"],
                                        data_params["n_samples"])
    input_test, target_test, _ = data.factory(data_params["dataset"],
                                              100,
                                              data_params["n_samples"])

    mdl = rkm.RKM(cuda=params["cuda"])
    mdl.append_level(**params["level1"])
    mdl.append_level(**params["level2"])
    mdl.append_level(**params["level3"])
    mdl.learn(input, target, maxiter=5e+3, tol=1e-6)
    y_hat = mdl.evaluate(input_test)
    MSE = np.mean((y_hat-target_test)**2)
    print(MSE)

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
