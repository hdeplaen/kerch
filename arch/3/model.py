import rkm
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import time
from sklearn import metrics
import random


class KernelNetwork(nn.Module):
    @rkm.kwargs_decorator({'kernel_type': 'expkernel',
                       'sigma': 1.,
                       'cuda': True,
                       'plot': False,
                       'range': (-1, 1, -1, 1),
                       'gamma': .1,
                       'aggregate': False,
                       'sigma_trainable': False,
                       'points_trainable': False,
                       'tanh': False})
    def __init__(self, num_data, num_kernels, num_classes, **kwargs):
        super(KernelNetwork, self).__init__()

        # MODEL PARAMETERS
        self.num_data = num_data
        self.num_kernels = num_kernels
        self.num_classes = num_classes
        self.kernel_type = kwargs['kernel_type']
        self.sigma = kwargs['sigma']
        self.gamma = kwargs['gamma']
        self.do_tanh = kwargs['tanh']

        # TRAINING PARAMETERS
        self.sigma_trainable = kwargs['sigma_trainable']
        self.points_trainable = kwargs['points_trainable']
        self.do_aggregate = kwargs['aggregate']

        # PLOT PARAMS
        self.do_plot = kwargs['plot']
        self.range = kwargs['range']
        self.xmin = self.range[0]
        self.xmax = self.range[1]
        self.ymin = self.range[2]
        self.ymax = self.range[3]

        # CUDA
        self.cuda = kwargs['cuda']
        if torch.cuda.is_available() and self.cuda:
            self.device = torch.device('cuda')
            torch.cuda.empty_cache()
            print('Using CUDA')
        else:
            self.device = torch.device('cpu')
            self.cuda = False

        # MODEL
        self.criterion = nn.MSELoss(reduction='mean')  # 'sum' also possible
        self.model = nn.ModuleDict({
            'hilbert': rkm.Hilbert(self.num_data,
                               self.num_kernels,
                               self.num_classes,
                               self.sigma,
                               kernel_type=self.kernel_type,
                               sigma_trainable=self.sigma_trainable,
                               points_trainable=self.points_trainable)
            # 'tanh': nn.Tanh()
        })

        # TRAINING ALGORITHMS
        self.lr = 5e-2
        self.wd = 1e-3
        self.params_opt = 'hilbert'
        self.opt1 = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.current_loss = -1

    def forward(self, x, idx_sv=None):
        x = self.model['hilbert'](x, idx_sv)
        if self.do_tanh: x = self.model['tanh'](3 * x)
        return x

    def evaluate(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tilde = self.forward(x)
        return y_tilde.detach().cpu().numpy()

    def _loss(self, x, y, idx_sv):
        x_tilde = self.forward(x, idx_sv)
        c = .5 * self.criterion(x_tilde, y)

        if self.gamma is not 0:
            reg = .5 * self.model['hilbert'].regularization_term(idx_sv)
            return self.gamma * c + reg, c, reg
        else:
            return c, c, 0

    def custom_train(self, x, y, max_iter=50000, min_cost=0, sz_sv=0):
        if sz_sv==0: sz_sv=self.num_kernels

        # LSSVM
        alpha, beta, loss, recon, reg = rkm.lssvm.dual(x, y, self.sigma, self.gamma)
        print("LSSVM")
        print("Bias: %f" % beta)
        print("Reconstruction: %f" % recon)
        print("Regularization %f" % reg)

        # PLOTS
        self.plt = rkm.plot(self,
                            xmin=self.xmin,
                            xmax=self.xmax,
                            ymin=self.ymin,
                            ymax=self.ymax,
                            cuda=self.cuda,
                            x=x, y=y,
                            lssvm_alpha=alpha,
                            kernel_type=self.kernel_type)

        # PUT MODEL IN MEMORY
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self._special_init(x, y)
        self.to(self.device)

        # self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # PROJECT ALPHA ONTO HYPERPLANE
        def correct_alpha():
            alpha = self.model['hilbert'].model['lin'].weight.data
            alpha -= torch.mean(alpha)
            self.model['hilbert'].model['lin'].weight.data = alpha

        # CLOSURE
        def closure():
            self.opt.zero_grad()
            idx_sv = random.choices(range(self.num_kernels), k=sz_sv)
            # y_tilde = self.forward(x, idx_sv)
            # idx_sv = None
            loss, recon, reg = self._loss(x, y, idx_sv)
            self.current_loss = loss.item()
            self.recon = recon.item()
            self.reg = reg.item()
            loss.backward(create_graph=True)
            return loss #, y_tilde

        # TRAINING LOOP
        t = trange(max_iter, desc='ML')
        self.t_start = time.time()
        for epoch in t:
            self.opt.zero_grad()
            self.opt.step(closure)
            correct_alpha()

            t.set_description('ML (recon=%g, reg=%g)' % (self.recon, self.reg))

            if (epoch % 250) == 0:
                if self.do_aggregate: self._aggregate()
                if self.do_plot: self.plt.imgen()

        self.cpu()

        if self.do_plot:
            self.plt.gifgen()

    def _special_init(self, x, y):
        # idx = torch.randperm(len(x))[:self.num_kernels]
        # x = x[idx,:]
        # y = y[idx]

        self.model['hilbert'].model[self.kernel_type]._special_init(x)
        self.model['hilbert'].model['lin']._special_init(y)

    def _aggregate(self, tol_dist=1e-1, tol_alpha=5e-2):
        # MERGE
        def merge(tol_dist):
            p = self.model['hilbert'].model[self.kernel_type].param.detach().cpu().numpy().transpose()
            D = np.triu(metrics.pairwise_distances(p), k=1)
            mask = np.logical_and(D < tol_dist, D > 0)
            idx = np.argwhere(mask)
            num_fusions = idx.shape[0]

            def merge_points(idx1, idx2):
                self.num_kernels -= 1
                self.model['hilbert'].num_kernels -= 1
                self.model['hilbert'].model[self.kernel_type].merge_kernels(idx1, idx2)
                self.model['hilbert'].model['lin'].merge_kernels(idx1, idx2)

            for i in range(num_fusions):
                idx1 = idx[i, 0]
                idx2 = idx[i, 1]
                idx[idx == idx2] = idx1
                idx[idx > idx2] -= 1
                merge_points(idx1, idx2)

        # SUPPRESS ALPHA
        def suppress(tol_alpha):
            w = self.model['hilbert'].model['lin'].weight.data
            max_w = w.abs().max()
            cond = torch.abs(w) / max_w < tol_alpha
            idx = np.argwhere(cond.cpu())
            num_suppressions = idx.shape[1]

            def suppress_point(idx_loc):
                self.num_kernels -= 1
                self.model['hilbert'].num_kernels -= 1
                self.model['hilbert'].model[self.kernel_type].suppress_kernel(idx_loc)
                self.model['hilbert'].model['lin'].suppress_kernel(idx_loc)

            for i in range(num_suppressions):
                idx_loc = idx[0, i]
                suppress_point(idx_loc)
                idx[idx > idx_loc] -= 1

        merge(tol_dist)
        suppress(tol_alpha)

        # UPDATE OPTIMIZERS
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # self.opt2 = HessianFree(self.model.parameters(), use_gnm=True, verbose=False)
        self.model['hilbert'].zero_grad()