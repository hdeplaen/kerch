"""
Runs RKM models.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: April 2021
"""

import argparse
import rkm.expes.run as run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs RKM models.')
    parser.add_argument('--experiment',
                        default=2,
                        type=int,
                        help='Experiment number (1: lssvm, 2: kpca)')

    args = parser.parse_args()
    experiment = {1: lambda: run.lssvm(),
                  2: lambda: run.kpca()}
    experiment.get(args.experiment, "Invalid experiment number.")()