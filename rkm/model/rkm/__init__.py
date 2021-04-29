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

        self.__model = []
        self.__euclidean = torch.nn.ParameterList()
        self.__stiefel = torch.nn.ParameterList()

    def __str__(self):
        text = f"This RKM has {len(self.__model)} levels:\n"
        for level in self.__model:
            text += (level.__str__() + "\n")
        return text

    def level(self, num=None):
        assert num is not None, "Layer number must be specified."
        assert num > 0, "Levels start at number 1."
        assert num <= len(self.__model), "Model has less levels than requested."
        return self.__model[num + 1]

    def __optstep(self, opt, closure):
        """
        One optimization step
        :param closure: closure function as in a torch optim step call.
        """
        # opt steps
        opt.step()

        for level in self.__model:
            level.after_step()

    def forward(self, x):
        for level in self.__model:
            x = level(x)
        return x

    def loss(self, x, y):
        tot_loss = 0.

        for level in self.__model:
            if level.eta == 0.:
                tot_loss += level.loss(x, y)
                x = level(x)

        return tot_loss

    @rkm.kwargs_decorator({"type": "sgd"})
    def learn(self, x, y, maxiter=int(1e+3), **kwargs):
        """

        :param x: input
        :param y: desired output
        :param maxiter: maximum number of iteration (default 1000)
        :param kwargs: optimizer parameters (default SGD with learning rate of 0.001)
        """

        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)
        self.to(self.device)
        opt = Optimizer.Optimizer(self.__euclidean,
                            self.__stiefel,
                            type=kwargs["type"],
                            **kwargs)

        def closure():
            opt.zero_grad()
            l = self.loss(x, y)
            l.backward(create_graph=True)
            return l

        t = trange(maxiter, desc='Training model')
        for iter in range(maxiter):
            self.__optstep(opt, closure)

    def evaluate(self, x):
        x.to(self.device)
        self.to(self.device)
        return self.forward(x).detach().cpu()

    @rkm.kwargs_decorator({"size_in": 1})
    def append_level(self, type, constraint="soft", **kwargs):
        """

        :param type: 'lssvm' or 'kpca'
        :param constraint: 'hard' or 'soft'
        :param kwargs: layer parameters
        """
        current = len(self.__model)
        size_in_new = kwargs["size_in"]
        size_out_old = self.__model[current].size_out

        assert size_in_new == size_out_old, \
            "Layer " + str(current + 1) + " (size_in=" + str(kwargs["size_in"]) + \
            ") not compatible with layer " + str(current) + " (size_out=" + str() + ")."

        switcher1 = {"lssvm": SoftLSSVM.SoftLSSVM(**kwargs),
                     "kpca": SoftKPCA.SoftKPCA(**kwargs)}
        switcher2 = {"lssvm": HardLSSVM.HardLSSVM(**kwargs),
                     "kpca": HardKPCA.HardKPCA(**kwargs)}
        switcher3 = {"soft": switcher1, "hard": switcher2}
        level = switcher3.get(constraint, "Invalid level contraint (soft/hard)"). \
            get(type, "Invalid level type (kpca/lssvm)")

        euclidean, stiefel = level.get_params()
        self.__euclidean.append(euclidean)
        self.__stiefel.append(stiefel)
        self.__model.append(level)
