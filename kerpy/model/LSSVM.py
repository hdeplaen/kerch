

from ..rkm import lssvm
from .Model import Model

class LSSVM(lssvm, Model):
    def __init__(self, **kwargs):
            super(LSSVM, self).__init__(**kwargs)


