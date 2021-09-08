"""
Base RKM model and various implementations

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
import numpy as np
from tqdm import trange

import rkm
import rkm.model.opt as Optimizer
import rkm.model.lssvm.SoftLSSVM as SoftLSSVM
import rkm.model.lssvm.HardLSSVM as HardLSSVM
import rkm.model.kpca.SoftKPCA as SoftKPCA
import rkm.model.kpca.HardKPCA as HardKPCA
import rkm.plot as rkmplot


class RKM(torch.nn.Module):
    @rkm.kwargs_decorator(
        {"cuda": False})
    def __init__(self, **kwargs):
        super(RKM, self).__init__()

        # CUDA
        self.cuda = kwargs["cuda"]
        if torch.cuda.is_available() and self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
            print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            self.cuda = False
            print("Using CPU")

        self._model = torch.nn.ModuleList()
        self._euclidean = torch.nn.ParameterList()
        self._slow = torch.nn.ParameterList()
        self._stiefel = torch.nn.ParameterList()

        self._last_loss = torch.tensor(0.)

    def __str__(self):
        text = f"RKM with {len(self._model)} levels:\n"
        num = 1
        for level in self._model:
            text += ("LEVEL" + str(num) + ": " + level.__str__() + "\n")
            num += 1
        return text

    @property
    def model(self):
        return self._model

    @property
    def num_levels(self):
        return len(self.model)

    @property
    def last_loss(self):
        return self._last_loss.data

    def level(self, num=None):
        assert num is not None, "Layer number must be specified."
        assert num >= 0, "Levels start at number 0."
        assert num < len(self.model), "Model has less levels than requested."
        return self.model[num]

    def _optstep(self, opt, closure):
        """
        One optimization step
        :param closure: closure function as in a torch optim step call.
        """
        for level in self.model: level.reset()
        opt.step(closure)
        for level in self.model: level.projection()

    def forward(self, x):
        for level in self.model:
            x = level(x)
        return x

    def loss(self, x, y):
        tot_loss = 0.

        for level in self.model:
            l, x = level.loss(x, y)
            tot_loss += level.eta * l

        self._last_loss = tot_loss
        return tot_loss

    @rkm.kwargs_decorator(
        {"maxiter": 1e+3,
         "tol": 1e-5,
         "epoch": 100,
         "early_stopping": 3,
         "stochastic": 1.})
    def learn(self, x, y, verbose=False, val_x=None, val_y=None, test_x=None, test_y=None, **kwargs):
        """

        :param x: input
        :param y: desired output
        :param maxiter: maximum number of iteration (default 1000)
        :param kwargs: optimizer parameters (default SGD with learning rate of 0.001)
        """
        maxiter = int(kwargs["maxiter"])
        tol = kwargs["tol"]
        epoch = kwargs["epoch"]
        early_stopping = kwargs["early_stopping"]
        stochastic = kwargs["stochastic"]
        test = test_x is not None and test_y is not None
        val = val_x is not None and val_y is not None
        if test and val:
            val_x = torch.tensor(val_x, dtype=rkm.ftype).to(self.device)
            val_y = torch.tensor(val_y, dtype=rkm.ftype).to(self.device)
            test_x = torch.tensor(test_x, dtype=rkm.ftype).to(self.device)
            test_y = torch.tensor(test_y, dtype=rkm.ftype).to(self.device)
        tr_text = "empty"
        val_text = "empty"
        test_text = "empty"

        x = torch.tensor(x, dtype=rkm.ftype).to(self.device)
        y = torch.tensor(y, dtype=rkm.ftype).to(self.device)
        self.to(self.device)
        opt = Optimizer.Optimizer(self._euclidean,
                                  self._slow,
                                  self._stiefel,
                                  **kwargs)

        plotenv = rkmplot.plotenv(model=self, opt=opt)

        def closure():
            opt.zero_grad()

            if stochastic < 1.:
                idx_stoch = random.sample(range(x.shape[0]), k=np.minimum(
                    int(self._stochastic * self._init_kernels), self.num_kernels))

            loss = self.loss(x, y)
            loss.backward(create_graph=True)
            return loss

        min_loss = float("Inf")

        tr = trange(int(maxiter), desc='Training model')

        if val: best_tr, best_val = 100, 100
        if val and test: best_test = 100
        early_stopping_count = 0

        for iter in tr:
            if self.cuda:
                torch.cuda.empty_cache()

            self._optstep(opt, closure)
            current_loss = self.last_loss

            # EPOCH
            if (iter % epoch) == 0 and test and val:
                tr.set_description(f"Loss: {current_loss:6.4e}, Tr:{tr_text}, V:{val_text}, Te:{test_text}, ES:{early_stopping_count}/{early_stopping}")
                if abs(current_loss - min_loss) < tol: break

                tr_mse = torch.mean((self.evaluate(x, numpy=False) - y) ** 2) * 100
                tr_text = f"{tr_mse:4.2f}%"

                val_mse = torch.mean((self.evaluate(val_x, numpy=False) - val_y) ** 2) * 100
                val_text = f"{val_mse:4.2f}%"

                test_mse = torch.mean((self.evaluate(test_x, numpy=False) - test_y) ** 2) * 100
                test_text = f"{test_mse:4.2f}%"

                # EARLY STOPPING
                if val_mse < best_val:
                    best_tr = tr_mse
                    best_val = val_mse
                    best_test = test_mse
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    if early_stopping_count > early_stopping:
                        break

                plotenv.update(iter, tr_mse=tr_mse, val_mse=val_mse, test_mse=test_mse, es=early_stopping_count)
                if verbose: print(self)

            if current_loss < min_loss: min_loss = current_loss

        plotenv.finish(best_tr=best_tr, best_val=best_val, best_test=best_test)
        if val: print(f"Best validation: {best_val:4.2f}%")
        if val and test: print(f"Corresponding test: {best_test:4.2f}%")

    def evaluate(self, x, numpy=True):
        if numpy: x = torch.tensor(x, dtype=rkm.ftype).to(self.device)
        self.to(self.device)
        for level in self.model:
            x = level.evaluate(x)
        if numpy: x = x.detach().cpu().numpy()
        return x

    @rkm.kwargs_decorator({"size_in": 1, "constraint": "soft", "representation": "dual"})
    def append_level(self, type, **kwargs):
        """

        :param type: 'lssvm' or 'kpca'
        :param constraint: 'hard' or 'soft'
        :param kwargs: layer parameters
        """
        current = len(self.model)
        size_in_new = kwargs["size_in"]

        if current > 0:
            size_out_old = self.model[current - 1].size_out
            assert size_in_new == size_out_old, \
                "Layer " + str(current + 1) + " (size_in=" + str(kwargs["size_in"]) + \
                ") not compatible with layer " + str(current) + " (size_out=" + str() + ")."

        # TO DO: make switchers cleaner with proper NameError
        switcher2 = {"soft": SoftKPCA.SoftKPCA, "hard": HardKPCA.HardKPCA}
        switcher3 = {"soft": SoftLSSVM.SoftLSSVM, "hard": HardLSSVM.HardLSSVM}
        switcher4 = {"soft": torch.nn.Sigmoid, "hard": torch.nn.Hardsigmoid}
        switcher1 = {"kpca": switcher2, "lssvm": switcher3, "sigmoid": switcher4}

        level = switcher1.get(type, "Invalid level type (kpca/lssvm)"). \
            get(kwargs["constraint"], "Invalid level constraint (soft/hard)")(device=self.device, **kwargs)

        euclidean, slow, stiefel = level.get_params(slow_names='sigma')
        self._euclidean.extend(euclidean)
        self._slow.extend(slow)
        self._stiefel.extend(stiefel)
        self._model.append(level)
