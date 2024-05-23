"""
Plotting solutions for a deep RKM _src.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

import kerch._archive.plot.plotenv_parent as plotenv_parent
import os

import kerch._archive.model.level.PrimalLinear as PrimalLinear
import kerch._archive.model.level.DualLinear as DualLinear
from kerch._archive import add_dict

import wandb

class plotenv_wandb(plotenv_parent.plotenv_parent):
    def __init__(self, model: rkm, opt: opt.Optimizer):
        super(plotenv_wandb, self).__init__(model, opt)
        os.environ["WANDB_SILENT"] = "true"
        self.SAVE_ALL = False
        print('Loading Weights and Biases...')

        ## LOGDIR
        name, dir, id = self.names
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.run = wandb.init(name=name,
                              dir=dir,
                              id=id,
                              project='RKM',
                              entity="hdeplaen",
                              reinit=True)
        self.artifact = wandb.Artifact(name=id,
                                       type='_src')

        self._hyperparameters()

    def _hyperparameters(self):
        hparams_dict = self.opt.hparams
        for num in range(self.model.num_levels):
            name = f"LEVEL{num}"
            level = self.model.Level(num)
            level_dict = {name + str(key): val for key, val in level.hparams.items()}
            hparams_dict = {**hparams_dict, **level_dict}

        self.run.config.update(hparams_dict)

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        self.run.log({"Total Loss": self.model.last_loss}, step=iter)
        self.run.log({"Early Stopping": es}, step=iter)

        for num in range(self.model.num_levels):
            level = self.model.Level(num)
            self.run.log({f"LEVEL{num} Loss": level.last_loss}, step=iter) # Level losses
            dict =  add_dict(f"LEVEL{num}", level.kernel.params)
            wandb.log(dict, step=iter)
            if isinstance(level, kpca.KPCA):
                self.kpca(num, level, iter)
            elif isinstance(level, lssvm.LSSVM):
                self.lssvm(num, level, iter)
            else:
                print(f"LEVEL{num} not recognized and cannot be plotted.")

        if val_mse is not None:
            self.run.log({'Validation Error': val_mse}, step=iter)
        if test_mse is not None:
            self.run.log({'Testing Error': test_mse}, step=iter)
        if tr_mse is not None:
            self.run.log({'Training Error': tr_mse}, step=iter)

    def kpca(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            # P = Level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            # P = Level.linear.weight
            K, _ = level.kernel.pmatrix()
        else:
            K = []

        wandb.log({f"LEVEL{num} (Kernel)", wandb.Image(K.data)}, step=iter)
        # wandb.log({f"LEVEL{num} (Projector)", P}, step=iter)

    def lssvm(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight
            K, _ = level.kernel.pmatrix()
        else:
            P = []
            K = []

        self.run.log({f"LEVEL{num} (Regularization term)": level.last_reg}, step=iter)
        self.run.log({f"LEVEL{num} (Reconstruction term)": level.last_recon}, step=iter)

        self.run.log({f"LEVEL{num} (Kernel)": wandb.Image(K.data)}, step=iter)
        self.run.log({f"LEVEL{num} (Support Vector Values)": wandb.Histogram(P.data)}, step=iter)

    def save_model(self, iter, best_tr, best_val, best_test):
        path = super(plotenv_wandb, self).save_model(iter, best_tr, best_val, best_test)
        if self.SAVE_ALL:
            self.artifact.add_file(path, is_tmp=True)

        best = {"Best Training Error": best_tr}
        if best_val is not None:
            best = {"Best Validation Error": best_val, **best}
        if best_test is not None:
            best = {"Best Test Error": best_test, **best}
        self.run.log(best, step=iter)
        self.run.config.update(best, allow_val_change=True)

    def finish(self, best_tr, best_val, best_test):
        path = super(plotenv_wandb, self).finish(best_tr, best_val, best_test)
        print('Updating to Weights and Biases...')
        self.artifact.add_file(path)
        self.run.log_artifact(self.artifact)
        self.run.finish()