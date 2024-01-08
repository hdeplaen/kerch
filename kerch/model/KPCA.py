from ._Model import _Model

class KPCA(_Model):
    def __init__(self, *args, **kwargs):
        super(KPCA, self).__init__(*args, **kwargs)
        kwargs['level_type'] = 'kpca'
        self._add_level(*args, **kwargs)