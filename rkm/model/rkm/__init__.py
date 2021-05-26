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

        self._model = torch.nn.ModuleList()
        self._euclidean = torch.nn.ParameterList()
        self._stiefel = torch.nn.ParameterList()

    def __str__(self):
        text = f"This RKM has {len(self._model)} levels:\n"
        for level in self._model:
            text += (level.__str__() + "\n")
        return text

    def level(self, num=None):
        assert num is not None, "Layer number must be specified."
        assert num > 0, "Levels start at number 1."
        assert num <= len(self._model), "Model has less levels than requested."
        return self._model[num + 1]

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

        return tot_loss

    def learn(self, x, y, maxiter=int(5e+4), tol=1e-9, **kwargs):
        """

        :param x: input
        :param y: desired output
        :param maxiter: maximum number of iteration (default 1000)
        :param kwargs: optimizer parameters (default SGD with learning rate of 0.001)
        """

        x = torch.tensor(x, dtype=rkm.ftype).to(self.device)
        y = torch.tensor(y, dtype=rkm.ftype).to(self.device)
        self.to(self.device)
        opt = Optimizer.Optimizer(self._euclidean,
                                  self._stiefel,
                                  **kwargs)

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
        tr = trange(maxiter, desc='Training model')
        for iter in tr:
            current_loss = self._optstep(opt, closure, solve).data

            if (iter % 10) == 0:
                tr.set_description(f'Loss: {current_loss}')
                if abs(current_loss - min_loss) < tol: break

            if current_loss < min_loss: min_loss = current_loss

        print('Learning model completed.')

    def evaluate(self, x):
        x.to(self.device)
        self.to(self.device)
        return self.forward(x).detach().cpu()

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
            size_out_old = self._model[current].size_out
            assert size_in_new == size_out_old, \
                "Layer " + str(current + 1) + " (size_in=" + str(kwargs["size_in"]) + \
                ") not compatible with layer " + str(current) + " (size_out=" + str() + ")."

        # TO DO: make switchers cleaner with proper NameError
        switcher1 = {"lssvm": SoftLSSVM.SoftLSSVM, "kpca": SoftKPCA.SoftKPCA}
        switcher2 = {"lssvm": HardLSSVM.HardLSSVM, "kpca": HardKPCA.HardKPCA}
        switcher3 = {"soft": switcher1, "hard": switcher2}

        level = switcher3.get(kwargs["constraint"], "Invalid level constraint (soft/hard)"). \
            get(type, "Invalid level type (kpca/lssvm)")(device=self.device, **kwargs)

        euclidean, stiefel = level.get_params()
        self._euclidean.extend(euclidean)
        self._stiefel.extend(stiefel)
        self._model.append(level)
