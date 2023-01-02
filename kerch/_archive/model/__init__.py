"""
Source code for the RKM toolbox.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

class ImplicitError(Exception):
    def __init__(self):
        self.message = "Dual representation not available."
        super(ImplicitError, self).__init__(self.message)


class ExplicitError(Exception):
    def __init__(self):
        self.message = "Primal representation not available. \
        The explicit representation lies in an infinite dimensional Hilbert space."
        super(ExplicitError, self).__init__(self.message)


class RepresentationError(Exception):
    def __init__(self):
        self.message = "Unrecognized or unspecified representation (must be primal or dual)."
        super(RepresentationError, self).__init__(self.message)