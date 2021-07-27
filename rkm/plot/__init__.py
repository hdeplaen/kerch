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
import rkm.model.level.PrimalLinear as PrimalLinear
import rkm.model.level.DualLinear as DualLinear

class plotenv():
    def __init__(self, model: RKM):
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        self.model = model
        self.trs = np.array([])
        self.vals = np.array([])
        self.tests = np.array([])

    def update(self, tr_mse=None, val_mse=None, test_mse=None) -> None:
        for num in range(self.model.num_levels):
            level = self.model.level(num)
            if isinstance(level, KPCA.KPCA):
                self.kpca(num, level)
            elif isinstance(level, LSSVM.LSSVM):
                self.lssvm(num, level)
            else:
                print(f"LEVEL{num} not recognized and cannot be plotted.")

        if val_mse is not None:
            np.append(self.vals, val_mse.data.detach().cpu().numpy())
        if test_mse is not None:
            np.append(self.tests, test_mse.data.detach().cpu().numpy())
        if tr_mse is not None:
            np.append(self.trs, tr_mse.data.detach().cpu().numpy())
            self.losses()

    def kpca(self, num, level):
        #assert(isinstance(level, KPCA.KPCA), "This level is not an instance of KPCA and cannot be plotted as such.")
        fig = plt.figure(num)
        axes = fig.subplots(1, 2)

        #TITLE
        t1 = level.__str__() + "\n"
        plt.title(t1)

        #SUBPLOTS
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha.data.detach().cpu().numpy()
            K = level.kernel.dmatrix().data.detach().cpu().numpy()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight.data.detach().cpu().numpy()
            K = level.kernel.pmatrix().data.detach().cpu().numpy()
        else:
            P = np.array([])
            K = np.array([])

        axes[0].imshow(K)
        axes[0].set_title("Correlation or kernel matrix")
        axes[1].imshow(P)
        axes[1].set_title("Projector")


    def lssvm(self, num, level):
        #assert(isinstance(level, LSSVM.LSSVM), "This level is not an instance of LSSVM and cannot be plotted as such.")
        fig = plt.figure(num)
        axes = fig.subplots(1, 2)

        # TITLE
        t1 = level.__str__() + "\n"
        t2 = f"Bias: {level.linear.bias.data.detach().cpu().numpy()}"
        plt.title(t1 + t2)

        # SUBPLOTS
        if isinstance(level.linear, DualLinear):
            P = level.linear.alpha.data.detach().cpu().numpy()
            K = level.kernel.dmatrix().data.detach().cpu().numpy()
        elif isinstance(level.linear, PrimalLinear):
            P = level.linear.weight.data.detach().cpu().numpy()
            K = level.kernel.pmatrix().data.detach().cpu().numpy()
        else:
            P = np.array([])
            K = np.array([])

        ##
        axes[0].imshow(K)
        axes[0].set_title("Correlation or kernel matrix")

        ##
        P = P.squeeze()
        # assert P.size[1] <= 1, "Plotting not implemented for multiple outputs now."
        axes[1].set_xlim((-1, level.init_kernels))
        #axes[1].bar(range(len(P)), np.abs(P), colors=np.where(P >= 0, 'g', 'r').squeeze())
        axes[1].bar(range(len(P)), P)
        axes[1].set_title("Weights or support vector values")

    def losses(self):
        plt.figure(0)
        #plt.plot(np.concatenate())

    def show(self):
        plt.show()

    def save(self):
        pass

    def gif(self):
        pass