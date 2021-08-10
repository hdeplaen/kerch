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

    parser = argparse.ArgumentParser(description='Runs RKM models.')
    parser.add_argument('--experiment',
                        default=3,
                        type=int,
                        help='Experiment number (1: lssvm, 2: kpca, 3: pima indians)')

    args = parser.parse_args()
    experiment = {1: run.lssvm,
                  2: run.kpca,
                  3: run.pima_indians}
    experiment.get(args.experiment, "Invalid experiment number.")()