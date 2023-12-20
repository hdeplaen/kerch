from .._PPCA import _PPCA
from .Level import Level

class WPCA(_PPCA, Level):
    def __init__(self, *args, **kwargs):
        super(WPCA, self).__init__(*args, **kwargs)