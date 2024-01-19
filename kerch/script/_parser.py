# coding=utf-8
import argparse
import yaml

from . import version

def get_args() -> dict:
    parser = argparse.ArgumentParser(description=f"Kernel methods and deep machines using PyTorch.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version',
                        action='version',
                        version=version.run(),
                        help='prints the version and other information')
    parser.add_argument('action',
                        metavar='action',
                        action='store',
                        type=str,
                        choices=['expe'],
                        nargs=1,
                        help="action to perform ('expe').")
    parser.add_argument('-f', '--filename',
                        action='store',
                        type=str,
                        help='path to the yaml file containing the experiment arguments',
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='prints the progress of the script')
    args = parser.parse_args()
    return vars(args)

def get_kwargs(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)