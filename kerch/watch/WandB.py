# coding=utf-8
from __future__ import annotations

import os
import wandb

from .Plotter import Plotter, Watcher
from ..utils import extend_docstring


@extend_docstring(Watcher)
class WandB(Plotter):
    def __init__(self, *args, **kwargs):
        super(WandB, self).__init__(*args, **kwargs)
        os.environ['WANDB_SILENT'] = "true"

        # INITIALIZE WANDB
        if not os.path.exists(self.dir_plotter):
            os.makedirs(self.dir_plotter)
        self._wandb_run = wandb.init(name=self.expe_name,
                                     dir=self.dir_plotter,
                                     id=self.expe_id,
                                     project='Kerch')
        self._wandb_run.define_metric(name='objective_loss', summary='min')
        self._wandb_run.define_metric(name='training_error', summary='min')
        self._wandb_run.define_metric(name='validation_error', summary='min')
        self._wandb_run.define_metric(name='test_error', summary='min')

        # LOG THE HYPERPARAMETERS
        model_hparams = self.model.hparams
        opt_hparams = {} if self.opt is None else self.opt.hparams
        hparams = {**opt_hparams, **model_hparams}
        self._wandb_run.config.update(hparams)

    @property
    def _plotter_name(self) -> str:
        return 'wandb'

    def finish(self) -> str:
        filepath = super(WandB, self).finish()
        self._wandb_run.log_model(path=filepath, name="final")
        self._wandb_run.finish(quiet=not self.verbose)
        return filepath

    def update(self,
               epoch: int,
               objective_loss: float,
               training_error: float | None = None,
               validation_error: float | None = None,
               test_error: float | None = None
               ) -> None:
        if epoch % self._num_epochs_save == 0:
            self.save_model(epoch)
        if epoch % self._num_epochs_params == 0:
            self._wandb_run.log(data=self.model.params, step=epoch, commit=False)
        if epoch % self._num_epochs_loss == 0:
            self._wandb_run.log(data=self.model.losses, step=epoch, commit=False)
            self._wandb_run.log(data={'objective_loss': objective_loss,
                                      'training_error': training_error,
                                      'validation_error': validation_error,
                                      'test_error': test_error}, step=epoch, commit=False)
        if epoch % self._num_epochs_plot == 0:
            self._wandb_run.log(data=self.model.watched_properties, step=epoch, commit=False)
        self._wandb_run.log(step=epoch)
