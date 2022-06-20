"""
Plotting solutions for a deep RKM _src.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

from torch.utils.tensorboard import SummaryWriter
import kerch._archive.plot.plotenv_parent as plotenv_parent
import os

import kerch._archive.model.level.PrimalLinear as PrimalLinear
import kerch._archive.model.level.DualLinear as DualLinear
from kerch._archive import invert_dict

class plotenv_tensorboard(plotenv_parent.plotenv_parent):
    def __init__(self, model: rkm, opt: opt.Optimizer):
        super(plotenv_tensorboard, self).__init__(model, opt)
        self.model = model

        ## LOGDIR
        log_dir = os.path.join(self.proj_dir, 'tensorboard', self.id)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.opt = opt

        # best = {"Training": 100,
        #         "Validation": 100,
        #         "Test": 100}
        # self._hyperparameters(best)

    def _hyperparameters(self, best):
        hparams_dict = self.opt.hparams
        for num in range(self.model.num_levels):
            name = f"LEVEL{num}"
            level = self.model.Level(num)
            level_dict = {name + str(key): val for key, val in level.hparams.items()}
            hparams_dict = {**hparams_dict, **level_dict}

        with self.writer as w:
            w.add_hparams(hparams_dict, best)

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        with self.writer as w:
            w.add_scalar("Total Loss", self.model.last_loss, global_step=iter)
            w.add_scalar("Early Stopping", es, global_step=iter)

            for num in range(self.model.num_levels):
                level = self.model.Level(num)
                w.add_scalars("Level Losses", {f"LEVEL{num}": level.last_loss}, global_step=iter)
                for (name, dict) in invert_dict(f"LEVEL{num}", level.kernel.params):
                    w.add_scalars(name, dict, global_step=iter)
                if isinstance(level, kpca.KPCA):
                    self.kpca(num, level, iter)
                elif isinstance(level, lssvm.LSSVM):
                    self.lssvm(num, level, iter)
                else:
                    print(f"LEVEL{num} not recognized and cannot be plotted.")

            if val_mse is not None:
                w.add_scalars('Error', {'Validating': val_mse}, global_step=iter)
            if test_mse is not None:
                w.add_scalars('Error', {'Testing': test_mse}, global_step=iter)
            if tr_mse is not None:
                w.add_scalars('Error', {'Training': tr_mse}, global_step=iter)

            w.flush()

    def kpca(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight
            K, _ = level.kernel.pmatrix()

        with self.writer as w:
            w.add_image(f"LEVEL{num} (Kernel)", K, global_step=iter, dataformats="HW")
            # w.add_image(f"LEVEL{num} (Projector)", P, global_step=iter, dataformats="HW")

    def lssvm(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight
            K, _ = level.kernel.pmatrix()

        with self.writer as w:
            self.writer.add_scalars("LSSVM Regularization Term", {f"LEVEL{num}": level.last_reg}, global_step=iter)
            self.writer.add_scalars("LSSVM Reconstruction Term", {f"LEVEL{num}": level.last_recon}, global_step=iter)

            self.writer.add_image(f"LEVEL{num} (Kernel)", K, global_step=iter, dataformats="HW")
            # self.writer.add_histogram(f"LEVEL{num} (Support Vector Values)", P, global_step=iter)

    def save_model(self, iter, best_tr, best_val, best_test):
        super(plotenv_tensorboard, self).save_model(iter, best_tr, best_val, best_test)

    def finish(self, best_tr, best_val, best_test):
        best = {"Training": best_tr}
        if best_val is not None:
            best = {"Validation": best_val, **best}
        if best_test is not None:
            best = {"Test": best_test, **best}
        self._hyperparameters(best)

        with self.writer as w:
            self.writer.flush()
            self.writer.close()