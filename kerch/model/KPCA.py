from .Model import Model

class KPCA(Model):
    def __init__(self, *args, **kwargs):
        super(KPCA, self).__init__(*args, **kwargs)
        kwargs['level_type'] = 'kpca'
        self._add_level(*args, **kwargs)