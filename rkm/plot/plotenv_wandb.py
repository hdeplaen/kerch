"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

import rkm.plot.plotenv_parent as plotenv_parent
import os

import rkm.model.rkm as RKM
import rkm.model.opt as OPT
import rkm.model.kpca as KPCA
import rkm.model.lssvm as LSSVM
import rkm.model.level.PrimalLinear as PrimalLinear
import rkm.model.level.DualLinear as DualLinear
from rkm.model.utils import invert_dict

import wandb

class plotenv_wandb(plotenv_parent.plotenv_parent):
    def __init__(self, model: RKM, opt: OPT.Optimizer):
        super(plotenv_wandb, self).__init__(model, opt)

        ## LOGDIR
        name, dir, id = self.names
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.run = wandb.init(name=name, dir=dir, id=id)
        self.artifact = wandb.Artifact(name=name, type='model')

        self._hyperparameters()

    def _hyperparameters(self):
        hparams_dict = self.opt.hparams
        for num in range(self.model.num_levels):
            name = f"LEVEL{num}"
            level = self.model.level(num)
            level_dict = {name + str(key): val for key, val in level.hparams.items()}
            hparams_dict = {**hparams_dict, **level_dict}

        self.run.config.update(hparams_dict)

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        self.run.log({"Total Loss": self.model.last_loss}, step=iter)
        self.run.log({"Early Stopping": es}, step=iter)

        for num in range(self.model.num_levels):
            level = self.model.level(num)
            self.run.log({f"LEVEL{num}": level.last_loss}, step=iter) # level losses
            for (name, dict) in invert_dict(f"LEVEL{num}", level.kernel.params):
                wandb.log(name, dict, step=iter)
            if isinstance(level, KPCA.KPCA):
                self.kpca(num, level, iter)
            elif isinstance(level, LSSVM.LSSVM):
                self.lssvm(num, level, iter)
            else:
                print(f"LEVEL{num} not recognized and cannot be plotted.")

        if val_mse is not None:
            self.run.log({'Validating': val_mse}, step=iter)
        if test_mse is not None:
            self.run.log({'Testing': test_mse}, step=iter)
        if tr_mse is not None:
            self.run.log({'Training': tr_mse}, step=iter)

    def kpca(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            # P = level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            # P = level.linear.weight
            K, _ = level.kernel.pmatrix()
        else:
            K = []

        wandb.log({f"LEVEL{num} (Kernel)", wandb.Image(K)}, step=iter)
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

        self.run.log({f"LEVEL{num} (Kernel)", wandb.Image(K)}, step=iter)
        self.run.log({f"LEVEL{num} (Support Vector Values)", wandb.Histogram(P)}, step=iter)

    def save_model(self, best_tr, best_val, best_test):
        path = super(plotenv_wandb, self).save_model(best_tr, best_val, best_test)
        self.artifact.add_file(path)
        self.run.log_artifact(self.artifact)

        best = {"Training": best_tr}
        if best_val is not None:
            best = {"Validation": best_val, **best}
        if best_test is not None:
            best = {"Test": best_test, **best}
        self.run.config.update(best)

    def finish(self, best_tr, best_val, best_test):
        self.run.finish()