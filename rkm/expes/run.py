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
    n_samples = params['n_samples']
    input, target, range = data.two_moons(n_samples)

    params = {'range': range}

    mdl = rkm.RKM(2, n_samples, 1, **params)
    mdl.custom_train(input, target, max_iter=int(2e+4), sz_sv=n_samples)


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
        content = yaml.load(file)
    return content.get(expe, 'Experiment not recognized in yaml file.')
