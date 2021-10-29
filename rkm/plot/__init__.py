"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: October 2021
"""

from abc import abstractmethod
import warnings
import tensorboard
import wandb

import rkm as rkm
import rkm.plot.plotenv_parent as plotenv_parent
import rkm.plot.plotenv_wandb as plotenv_wandb
import rkm.plot.plotenv_tensorboard as plotenv_tensorboard

import rkm.model.rkm as RKM
import rkm.model.opt as OPT

class plotenv(plotenv_parent.plotenv_parent):
    def __new__(cls, model: RKM, opt: OPT.Optimizer):
        if rkm.PLOT_ENV == 'wandb':
            return plotenv_wandb.plotenv_wandb(model, opt)
        elif rkm.PLOT_ENV == 'tensorboard':
            return plotenv_tensorboard.plotenv_tensorboard(model, opt)
        elif rkm.PLOT_ENV == 'both':
            pl = plotenv_tensorboard.plotenv_tensorboard(model, opt)
            name, log_dir, id = pl.names
            wandb.init(name=name,
                       dir=log_dir,
                       id=id,
                       sync_tensorboard=True,
                       project='RKM',
                       entity='hdeplaen',
                       reinit=True)
            return pl
        elif rkm.PLOT_ENV == 'none':
            return super(plotenv, cls).__init__(model, opt)
        else:
            warnings.warn('Plot environment not recognized. No plotting will occur.')
            return super(plotenv, cls).__init__(model, opt)