"""
Plotting solutions for a deep RKM _src.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: October 2021
"""

import warnings
import wandb

import kerch as rkm
import kerch._archive.plot.plotenv_parent as plotenv_parent
import kerch._archive.plot.plotenv_wandb as plotenv_wandb
import kerch._archive.plot.plotenv_tensorboard as plotenv_tensorboard


class plotenv(plotenv_parent.plotenv_parent):
    def __new__(cls, model: kerpy, opt: opt.Optimizer):
        if kerpy.PLOT_ENV == 'wandb':
            return plotenv_wandb.plotenv_wandb(model, opt)
        elif kerpy.PLOT_ENV == 'tensorboard':
            return plotenv_tensorboard.plotenv_tensorboard(model, opt)
        elif kerpy.PLOT_ENV == 'both':
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
        elif kerpy.PLOT_ENV == 'none':
            return super(plotenv, cls).__init__(model, opt)
        else:
            warnings.warn('Plot environment not recognized. No plotting will occur.')
            return super(plotenv, cls).__init__(model, opt)