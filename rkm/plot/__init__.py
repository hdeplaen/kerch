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
            self.vals = np.append(self.vals, val_mse.data.detach().cpu().numpy())
        if test_mse is not None:
            self.tests = np.append(self.tests, test_mse.data.detach().cpu().numpy())
        if tr_mse is not None:
            self.trs = np.append(self.trs, tr_mse.data.detach().cpu().numpy())
            self.losses()

    def kpca(self, num, level):
        #assert(isinstance(level, KPCA.KPCA), "This level is not an instance of KPCA and cannot be plotted as such.")
        fig = plt.figure(num+1)
        axes = fig.subplots(1, 2)

        #TITLE
        t0 = f"LEVEL {num + 1}" + "\n"
        t1 = level.__str__() + "\n"
        fig.suptitle(t0 + t1)

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

        axes[0].imshow(K, aspect='equal')
        axes[0].set_title("Correlation or kernel matrix")
        axes[1].imshow(P, aspect='auto')
        axes[1].set_title("Projector")

        plt.draw()
        plt.pause(0.01)
        plt.clf()


    def lssvm(self, num, level):
        #assert(isinstance(level, LSSVM.LSSVM), "This level is not an instance of LSSVM and cannot be plotted as such.")
        fig = plt.figure(num+1)
        axes = fig.subplots(1, 2)

        # TITLE
        t0 = f"LEVEL {num + 1}" + "\n"
        t1 = level.__str__() + "\n"
        t2 = f"Bias: {level.linear.bias.data.detach().cpu().numpy()}"
        fig.suptitle(t0 + t1 + t2)

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
        axes[1].bar(range(len(P)), np.abs(P), color=np.where(P >= 0, 'g', 'r').squeeze())
        #axes[1].bar(range(len(P)), P)
        axes[1].set_title("Weights or support vector values")

        plt.draw()
        plt.pause(0.01)
        plt.clf()

    def losses(self):
        plt.figure(0)
        x = range(0,50*len(self.trs),50)
        plt.plot(x, self.trs, label=f"Training {self.trs[-1]}")
        plt.plot(x, self.vals, label=f"Validation {self.vals[-1]}")
        plt.plot(x, self.tests, label=f"Test {self.tests[-1]}")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.legend()

        plt.draw()
        plt.pause(0.01)
        plt.clf()


    def show(self):
        pass
        # plt.draw()
        # plt.pause(0.01)
        # plt.clf()

    def save(self):
        pass

    def gif(self):
        pass