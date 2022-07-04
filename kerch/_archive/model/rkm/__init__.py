"""
Base RKM _src and various implementations

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from tqdm import trange

import kerch
from kerch.utils import process_y
import kerch._archive.model.level.IDXK as IDXK
import kerch.opt as Optimizer
import kerch._archive.model.lssvm.SoftLSSVM as SoftLSSVM
import kerch._archive.model.lssvm.HardLSSVM as HardLSSVM
import kerch._archive.model.kpca.SoftKPCA as SoftKPCA
import kerch._archive.model.kpca.HardKPCA as HardKPCA
import kerch._archive.plot as rkmplot


class RKM(torch.nn.Module):
    @kerch.kwargs_decorator(
        {"cuda": True,
         "verbose": True,
         "name":'noname'})
    def __init__(self, **kwargs):
        super(RKM, self).__init__()
        self._verbose = kwargs["verbose"]

        # CUDA
        self.cuda = kwargs["cuda"]
        if torch.cuda.is_available() and self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
            if self._verbose: print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            self.cuda = False
            if self._verbose: print("Using CPU")

        self._model = torch.nn.ModuleList()
        self._euclidean = torch.nn.ParameterList()
        self._slow = torch.nn.ParameterList()
        self._stiefel = torch.nn.ParameterList()

        self._last_loss = torch.tensor(0.)
        self._classifier = False

        self.name = kwargs['name']

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

    def init(self, x, y):
        for level in self.model:
            level.init(x, y)

    def cpu_forward(self, x, y):
        print('Initializing _src (CPU).')
        for level in self.model:
            level.kernels_init(x)
            x = level.forward(x, y, init=True)
            print('Level done.')
        print('Initializing done.')

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

    @kerch.kwargs_decorator(
        {"maxiter": 1e+3,
         "tol": 1e-5,
         "epoch": 1,
         "early_stopping": 3,
         "stochastic": 1.,
         "batches": 1,
         "init": False,
         "reduce_epochs": float('inf'),
         "reduce_rate": 2,
         "plot": True})
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
        batches = kwargs["batches"]
        init = kwargs["init"]
        reduce_epochs = kwargs["reduce_epochs"]
        reduce_rate = kwargs["reduce_rate"]
        plot = kwargs["plot"]

        test = test_x is not None and test_y is not None
        val = val_x is not None and val_y is not None

        if val:
            val_x = torch.tensor(val_x, dtype=kerch.ftype)
            val_y = torch.tensor(val_y, dtype=kerch.ftype)
            val_y = process_y(val_y)
        if test:
            test_x = torch.tensor(test_x, dtype=kerch.ftype)
            test_y = torch.tensor(test_y, dtype=kerch.ftype)
            test_y = process_y(test_y)

        val_error = None
        test_error = None
        tr_text = "empty"
        val_text = "empty"
        test_text = "empty"

        x = torch.tensor(x, dtype=kerch.ftype)
        y = torch.tensor(y, dtype=kerch.ftype)
        y = process_y(y)

        opt = Optimizer.Optimizer(self._euclidean,
                                  self._slow,
                                  self._stiefel,
                                  **kwargs)

        if plot:
            plotenv = rkmplot.plotenv(model=self, opt=opt)

        self.init(x, y)
        if init: self.cpu_forward(x, y)

        # LOADING ON DEVICE
        self.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        if val:
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)
        if test:
            test_x = test_x.to(self.device)
            test_y = test_y.to(self.device)

        self.idxk = None
        if stochastic < 1.:
            idx_kwargs = {'stochastic': stochastic, 'init_kernels': x.size(0), 'general': True}
            self.idxk = IDXK(**idx_kwargs)
            for level in self.model: level.init_idxk(self.idxk)

        min_loss = float("Inf")
        best_tr = 100
        if val:
            best_val = 100
        else:
            best_val = None
        if val and test:
            best_test = 100
        else:
            best_test = None
        early_stopping_count = 0

        def gen_batch(x, y):
            if self.idxk is not None:
                self.idxk.new_general()
                idx = self.idxk.idx_kernels
                x_loc = x[idx, :]
                y_loc = y[idx]
            else:
                x_loc = x
                y_loc = y
            return x_loc, y_loc

        with trange(int(maxiter), desc='Loss', position=0, leave=True, disable=not self._verbose) as tri:
            for iter in tri:
                if self.cuda:
                    torch.cuda.empty_cache()

                if self.idxk is not None:
                    self.idxk.new_general()

                def closure():
                    opt.zero_grad()
                    tot_loss = 0.

                    if self.idxk is not None:
                        for _ in trange(int(batches),
                                        desc=f'Batches using {self.idxk.num_samples}/{self.idxk.num_sample} datapoints',
                                        position=1,
                                        leave=False,
                                        disable=not self._verbose):
                            x_loc, y_loc = gen_batch(x, y)
                            loss = self.loss(x_loc, y_loc)
                            if opt.requires_grad: loss.backward(create_graph=False)
                            tot_loss += loss
                    else:
                        loss = self.loss(x, y)
                        if opt.requires_grad: loss.backward(create_graph=False)
                        tot_loss += loss
                    return tot_loss

                self._optstep(opt, closure)
                current_loss = self.last_loss

                # REDUCE LEARNING RATE
                if ((iter + 1) % reduce_epochs) == 0:
                    opt.reduce(rate=reduce_rate)

                # EPOCH
                if (iter % epoch) == 0:
                    if self._classifier:
                        tr_error = (1-torch.sum(torch.round(self.evaluate(x, numpy=False)) == y)/len(y)) * 100
                        if val: val_error = (1-torch.sum(torch.round(self.evaluate(val_x, numpy=False)) == val_y)/len(val_y)) * 100
                        if test: test_error = (1-torch.sum(torch.round(self.evaluate(test_x, numpy=False)) == test_y)/len(test_y)) * 100
                    else:
                        tr_error = torch.mean((self.evaluate(x, numpy=False) - y) ** 2) * 100
                        if val: val_error = torch.mean((self.evaluate(val_x, numpy=False) - val_y) ** 2) * 100
                        if test: test_error = torch.mean((self.evaluate(test_x, numpy=False) - test_y) ** 2) * 100

                    tri.set_description(
                        f"Loss: {current_loss:6.4e}, Tr:{tr_text}, V:{val_text}, Te:{test_text}, ES:{early_stopping_count}/{early_stopping}")

                    tr_text = f"{tr_error:4.2f}%({best_tr:4.2f}%)"
                    if val:
                        val_text = f"{val_error:4.2f}({best_val:4.2f}%)%"
                    else:
                        val_text = "N/A"
                    if test:
                        test_text = f"{test_error:4.2f}({best_test:4.2f}%)%"
                    else:
                        test_text = "N/A"

                    # CONVERGENCE
                    if abs(current_loss - min_loss) <= tol: break
                    if abs(current_loss) <= tol: break

                    # EARLY STOPPING
                    if val:
                        if val_error <= best_val:
                            best_tr = tr_error
                            best_val = val_error
                            if test: best_test = test_error
                            early_stopping_count = 0
                            if plot:
                                plotenv.save_model(iter,
                                                   best_tr=best_tr,
                                                   best_val=best_val,
                                                   best_test=best_test)
                        else:
                            early_stopping_count += 1
                            if early_stopping_count > early_stopping:
                                break

                    # OUTPUT
                    if plot: plotenv.update(iter,
                                            tr_mse=tr_error,
                                            val_mse=val_error,
                                            test_mse=test_error,
                                            es=early_stopping_count)

                if current_loss < min_loss: min_loss = current_loss

        if plot and not val: plotenv.save_model(iter,
                                                best_tr=best_tr,
                                                best_val=best_val,
                                                best_test=best_test)
        if plot: plotenv.finish(best_tr=best_tr,
                                best_val=best_val,
                                best_test=best_test)
        if val: print(f"\nBest validation: {best_val:4.2f}%")
        if val and test: print(f"Corresponding test: {best_test:4.2f}%")

        return self.evaluate(x, numpy=True).squeeze()

    def evaluate(self, x, numpy=True):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=kerch.ftype).to(self.device)
        self.to(self.device)
        for level in self.model:
            x = level.evaluate(x)
        if numpy: x = x.detach().cpu().numpy()
        return x

    @kerch.kwargs_decorator({"size_in": 1, "constraint": "soft", "representation": "dual"})
    def append_level(self, type, **kwargs):
        """

        :param type: 'LSSVM' or 'KPCA'
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
        switcher1 = {"KPCA": switcher2, "LSSVM": switcher3}

        level = switcher1.get(type, "Invalid Level type (KPCA/LSSVM)"). \
            get(kwargs["constraint"], "Invalid Level constraint (soft/hard)")(device=self.device, **kwargs)

        euclidean, slow, stiefel = level.get_params(slow_names='sigma')
        self._euclidean.extend(euclidean)
        self._slow.extend(slow)
        self._stiefel.extend(stiefel)
        self._model.append(level)
        self._classifier = level._classifier
