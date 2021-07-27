"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

import numpy as np
import matplotlib.pyplot as plt

import rkm.model.rkm as RKM
import rkm.model.kpca as KPCA
import rkm.model.lssvm as LSSVM

class plotenv():
    def __init__(self, model: RKM):
        self.model = model
        self.trs = np.empty()
        self.vals = np.empty()
        self.tests = np.empty()

    def update(self, tr_mse=None, val_mse=None, test_mse=None) -> None:
        for num in range(self.model.num_levels):
            level = self.model.level(num)
            if isinstance(level, KPCA):
                self.kpca(num, level)
            elif isinstance(level, LSSVM):
                self.lssvm(num, level)
            else:
                print(f"LEVEL{num} not recognized and cannot be plotted.")

        if val_mse is not None:
            self.vals.append(val_mse.data.detach().cpu().numpy())
        if test_mse is not None:
            self.tests.append(test_mse.data.detach().cpu().numpy())
        if tr_mse is not None:
            self.trs.append(tr_mse.data.detach().cpu().numpy())
            self.losses()

    def kpca(self, num, level):
        assert(isinstance(level, KPCA), "This level is not an instance of KPCA and cannot be plotted as such.")

    def lssvm(self, num, level):
        assert(isinstance(level, LSSVM), "This level is not an instance of LSSVM and cannot be plotted as such.")

    def losses(self):
        plt.figure(0)
        plt.plot(np.concatenate())

    def show(self):
        pass

    def save(self):
        pass

    def gif(self):
        pass