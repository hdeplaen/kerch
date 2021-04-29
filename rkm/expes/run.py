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

import rkm.expes.data as data
import rkm.model.rkm as rkm

def lssvm():
    params = load_params('lssvm')
    n_samples = params['n_samples']
    input, target, range = data.get("two_moons", n_samples)

    mdl = rkm.RKM(2, n_samples, 1, cuda=True)

    rkm.append_level(type="lssvm",
                  constraint="soft",
                  gamma=1.,
                  size_in=input.size(1),
                  init_kernels=input.size(0),
                  requires_bias=True,
                  kernel_type="rbf",
                  sigma=1.)

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
        content = yaml.load(file)
    return content.get(expe, 'Experiment not recognized in yaml file.')
