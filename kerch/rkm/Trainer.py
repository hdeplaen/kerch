import torch
import tqdm

from .RKM import RKM
from ..utils import kwargs_decorator

class Trainer():
    @kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model: RKM = kwargs["model"]

    def fit(self) -> None:
        self._model.init_sample()
        self._model.init_levels()
        self._model.init_targets()

        progress = tqdm.tqdm()

