from .Model import Model

class LSSVM(Model):
    def __init__(self, *args, **kwargs):
        super(LSSVM, self).__init__(*args, **kwargs)
        kwargs['level_type'] = 'lssvm'
        self._append_level(*args, **kwargs)

    def __str__(self):
        return "[Model] " + self._first_level.__repr__()