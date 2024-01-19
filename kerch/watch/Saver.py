# coding=utf-8
from __future__ import annotations

from .Watcher import Watcher
from ..utils import extend_docstring


@extend_docstring(Watcher)
class Saver(Watcher):
    def __init__(self, *args):
        super(Saver, self).__init__(*args)

    def update(self,
               epoch: int,
               objective_loss: float,
               training_error: float | None = None,
               validation_error: float | None = None,
               test_error: float | None = None
               ) -> None:
        if epoch % self._num_epochs_save == 0:
            self.save_model(epoch)
