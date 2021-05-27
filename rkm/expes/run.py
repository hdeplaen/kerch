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

from rkm.expes.data import data
import rkm.model.rkm as rkm

def lssvm():
    params = load_params('lssvm')
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
    params = load_params('kpca')
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


#######################################################################################################################

def load_params(expe: str):
    """
    Loads the parameters from the expe.yaml file.
    :param expe: string representing parameters.
    :return: dictionnary of parameters.
    """
    with open(os.path.join(sys.path[0], "expes.yaml"), "r") as file:
        content = yaml.safe_load(file)
    return content.get(expe, 'Experiment not recognized in yaml file.')
