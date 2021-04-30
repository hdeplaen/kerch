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

    mdl = rkm.RKM(cuda=params["cuda"])

    level_params = params["level"]
    mdl.append_level(**level_params)

    mdl.learn(input, target)

def kpca():
    pass


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
