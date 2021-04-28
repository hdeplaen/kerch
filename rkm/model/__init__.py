"""
Source code for the RKM toolbox.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.model.kernel as kernel
import rkm.model.kpca as kpca
import rkm.model.level as level
import rkm.model.lssvm as lssvm
import rkm.model.opt as opt
import rkm.model.rkm as rkm

class DualError(Exception):
    def __init__(self):
        self.message = "Dual representation not available."


class PrimalError(Exception):
    def __init__(self):
        self.message = "Primal representation not available. \
        The explicit representation lies in an infinite dimensional Hilbert space."


class RepresentationError(Exception):
    def __init__(self):
        self.message = "Unrecognized or unspecified representation (must be primal or dual)."