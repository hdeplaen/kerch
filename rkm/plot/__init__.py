"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

from torch.utils.tensorboard import SummaryWriter

import rkm.model.rkm as RKM
import rkm.model.kpca as KPCA
import rkm.model.lssvm as LSSVM
import rkm.model.level.PrimalLinear as PrimalLinear
import rkm.model.level.DualLinear as DualLinear
from rkm.model.utils import invert_dict

class plotenv():
    def __init__(self, model: RKM):
        self.model = model
        self.writer = SummaryWriter()
        self._hyperparameters()

    def _hyperparameters(self):
        pass

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None) -> None:
        self.writer.add_scalar("Total Loss", self.model.last_loss, global_step=iter)

        for num in range(self.model.num_levels):
            level = self.model.level(num)
            self.writer.add_scalars("Level Losses", {f"LEVEL{num}": level.last_loss}, global_step=iter)
            for (name, dict) in invert_dict(f"LEVEL{num}", level.kernel.params):
                self.writer.add_scalars(name, dict, global_step=iter)
            if isinstance(level, KPCA.KPCA):
                self.kpca(num, level, iter)
            elif isinstance(level, LSSVM.LSSVM):
                self.lssvm(num, level, iter)
            else:
                print(f"LEVEL{num} not recognized and cannot be plotted.")

        if val_mse is not None:
            self.writer.add_scalars('Error', {'Validating': val_mse}, global_step=iter)
        if test_mse is not None:
            self.writer.add_scalars('Error', {'Testing': test_mse}, global_step=iter)
        if tr_mse is not None:
            self.writer.add_scalars('Error', {'Training': tr_mse}, global_step=iter)

        self.writer.flush()

    def kpca(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight
            K = level.kernel.pmatrix()

        self.writer.add_image(f"LEVEL{num} (Kernel)", K, global_step=iter, dataformats="HW")
        self.writer.add_image(f"LEVEL{num} (Projector)", P, global_step=iter, dataformats="HW")

    def lssvm(self, num, level, iter):
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha
            K = level.kernel.dmatrix()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight
            K = level.kernel.pmatrix()

        self.writer.add_scalars("LSSVM Regularization Term", {f"LEVEL{num}": level.last_reg}, global_step=iter)
        self.writer.add_scalars("LSSVM Reconstruction Term", {f"LEVEL{num}": level.last_recon}, global_step=iter)

        self.writer.add_image(f"LEVEL{num} (Kernel)", K, global_step=iter, dataformats="HW")
        self.writer.add_histogram(f"LEVEL{num} (Support Vector Values)", P, global_step=iter)

    def finish(self):
        self.writer.flush()
        self.writer.close()

    def gif(self):
        pass