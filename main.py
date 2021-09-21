"""
Runs RKM models.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: April 2021
"""

import random
import argparse
import rkm.expes.run as run

if __name__ == "__main__":
    # random.seed(42)

    parser = argparse.ArgumentParser(description='Runs various deep RKM experiments.')
    parser.add_argument("-t", "--tests",
                        action='store_true',
                        default=False,
                        help="Performs a series of sanity checks as described in the sanity.yaml file.")
    parser.add_argument("-e", "--experiment",
                        default=None,
                        type=str,
                        help="Performs the experiment described in the expe.yaml file.")

    args = parser.parse_args()
    tests = args.tests
    experiment = args.experiment

    if tests: run.tests()
    if experiment is not None: run.experiment(experiment)