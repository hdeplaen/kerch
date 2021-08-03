"""
Base RKM model and various implementations

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
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
        text = f"\nThis RKM has {len(self._model)} levels:\n"
        num = 1
        for level in self._model:
            text += ("LEVEL" + str(num) + ": " + level.__str__() + "\n")
            num += 1
        return text

    @property
    def num_levels(self):
        return len(self._model)

    @property
    def last_loss(self):
        return self._last_loss.data

    def level(self, num=None):
        assert num is not None, "Layer number must be specified."
        assert num >= 0, "Levels start at number 0."
        assert num < len(self._model), "Model has less levels than requested."
        return self._model[num]

    def _optstep(self, opt, closure, solve):
        """
        One optimization step
        :param closure: closure function as in a torch optim step call.
        """
        loss = solve()

        for level in self._model:
            level.before_step()

        # opt steps
        opt.step(closure)

        for level in self._model:
            level.after_step()

        return loss

    def forward(self, x):
        for level in self._model:
            x = level(x)
        return x

    def loss(self, x, y):
        tot_loss = 0.

        for level in self._model:
            if level.eta != 0.:
                l, _ = level.loss(x, y)
                tot_loss += l

            level.layerin = x
            level.layerout = y
            x = level(x)

        self._last_loss = tot_loss.data
        return tot_loss

    @rkm.kwargs_decorator(
        {"maxiter": 1e+3,
         "tol": 1e-5,
         "step": 100})
    def learn(self, x, y, verbose=False, val_x=None, val_y=None, test_x=None, test_y=None, **kwargs):
        """

        :param x: input
        :param y: desired output
        :param maxiter: maximum number of iteration (default 1000)
        :param kwargs: optimizer parameters (default SGD with learning rate of 0.001)
        """
        maxiter = int(kwargs["maxiter"])
        tol = kwargs["tol"]
        step = kwargs["step"]
        test = test_x is not None and test_y is not None
        val = val_x is not None and val_y is not None
        if test and val:
            val_x = torch.tensor(val_x, dtype=rkm.ftype).to(self.device)
            val_y = torch.tensor(val_y, dtype=rkm.ftype).to(self.device)
            test_x = torch.tensor(test_x, dtype=rkm.ftype).to(self.device)
            test_y = torch.tensor(test_y, dtype=rkm.ftype).to(self.device)
        val_text = "empy"
        test_text = "empty"

        x = torch.tensor(x, dtype=rkm.ftype).to(self.device)
        y = torch.tensor(y, dtype=rkm.ftype).to(self.device)
        self.to(self.device)
        opt = Optimizer.Optimizer(self._euclidean,
                                  self._slow,
                                  self._stiefel,
                                  **kwargs)

        plotenv = rkmplot.plotenv(model=self, opt=opt)

        def solve():
            loss = self.loss(x, y)
            for level in self._model: level.before_step(x, y)
            return loss

        def closure():
            opt.zero_grad()
            loss = self.loss(x, y)
            loss.backward(create_graph=True)
            return loss

        min_loss = float("Inf")

        tr = trange(int(maxiter), desc='Training model')

        if val: best_tr, best_val = 100, 100
        if val and test: best_test = 100

        for iter in tr:
            current_loss = self._optstep(opt, closure, solve).data

            if (iter % 10) == 0 and verbose:
                tr.set_description(f"Loss: {current_loss:6.4e}, V{val_text}, T{test_text}")
                if abs(current_loss - min_loss) < tol: break

            if (iter % step) == 0 and test and val and verbose:
                tr_mse = torch.mean((self.evaluate(x, numpy=False) - y) ** 2) * 100
                tr_text = f"{tr_mse:4.2f}%"

                val_mse = torch.mean((self.evaluate(val_x, numpy=False) - val_y) ** 2) * 100
                val_text = f"{val_mse:4.2f}%"

                test_mse = torch.mean((self.evaluate(test_x, numpy=False) - test_y) ** 2) * 100
                test_text = f"{test_mse:4.2f}%"

                if val_mse < best_val:
                    best_tr = tr_mse
                    best_val = val_mse
                    best_test = test_mse

                plotenv.update(iter, tr_mse=tr_mse, val_mse=val_mse, test_mse=test_mse)
                if verbose: print(self)

            if current_loss < min_loss: min_loss = current_loss

        plotenv.finish(best_tr=best_tr, best_val=best_val, best_test=best_test)
        if val: print(f"Best validation: {best_val:4.2f}%\n")
        if val and test: print(f"Corresponding test: {best_test:4.2f}%\n")

    def evaluate(self, x, numpy=True):
        if numpy: x = torch.tensor(x, dtype=rkm.ftype).to(self.device)
        self.to(self.device)
        for level in self._model:
            x = level.evaluate(x)
        if numpy: x = x.detach().cpu().numpy()
        return x

    @rkm.kwargs_decorator({"size_in": 1, "constraint": "soft"})
    def append_level(self, type, **kwargs):
        """

        :param type: 'lssvm' or 'kpca'
        :param constraint: 'hard' or 'soft'
        :param kwargs: layer parameters
        """
        current = len(self._model)
        size_in_new = kwargs["size_in"]

        if current > 0:
            size_out_old = self._model[current - 1].size_out
            assert size_in_new == size_out_old, \
                "Layer " + str(current + 1) + " (size_in=" + str(kwargs["size_in"]) + \
                ") not compatible with layer " + str(current) + " (size_out=" + str() + ")."

        # TO DO: make switchers cleaner with proper NameError
        switcher1 = {"lssvm": SoftLSSVM.SoftLSSVM, "kpca": SoftKPCA.SoftKPCA}
        switcher2 = {"lssvm": HardLSSVM.HardLSSVM, "kpca": HardKPCA.HardKPCA}
        switcher3 = {"soft": switcher1, "hard": switcher2}

        level = switcher3.get(kwargs["constraint"], "Invalid level constraint (soft/hard)"). \
            get(type, "Invalid level type (kpca/lssvm)")(device=self.device, **kwargs)

        euclidean, slow, stiefel = level.get_params(slow_names='sigma')
        self._euclidean.extend(euclidean)
        self._slow.extend(slow)
        self._stiefel.extend(stiefel)
        self._model.append(level)
