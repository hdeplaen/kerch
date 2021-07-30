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

class plotenv():
    def __init__(self, model: RKM):
        self.model = model
        self.writer = SummaryWriter()

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None) -> None:
        for num in range(self.model.num_levels):
            level = self.model.level(num)
            self.writer.add_scalars(f"LEVEL{num}", level.kernel.params, global_step=iter)
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

        self.writer.add_image(f"LEVEL{num} (Kernel)", K, global_step=iter, dataformats="HW")
        self.writer.add_histogram(f"LEVEL{num} (Support Vector Values)", P, global_step=iter)

    def finish(self):
        self.writer.flush()
        self.writer.close()

    def gif(self):
        pass