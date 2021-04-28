from arch.kernel_utils import *
import torch
import torch.nn as nn
from tqdm import trange
from sklearn import metrics
import matplotlib.animation as animation
import time
from functools import wraps
from arch.lssvm_utils import LSSVM

# PLOTS
import matplotlib.pyplot as plt
import numpy as np


def kwargs_decorator(dict_kwargs):
    def wrapper(f):
        @wraps(f)
        def inner_wrapper(*args, **kwargs):
            new_kwargs = {**dict_kwargs, **kwargs}
            return f(*args, **new_kwargs)

        return inner_wrapper

    return wrapper


class KernelNetwork(nn.Module):
    @kwargs_decorator({'kernel_type': 'expkernel',
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
        self.criterion = nn.MSELoss(reduction='sum')  # 'mean' also possible
        self.model = nn.ModuleDict({
            'hilbert': Hilbert(self.num_data,
                               self.num_kernels,
                               self.num_classes,
                               self.sigma,
                               kernel_type=self.kernel_type,
                               sigma_trainable=self.sigma_trainable,
                               points_trainable=self.points_trainable)
            # 'tanh': nn.Tanh()
        })

        # TRAINING ALGORITHMS
        self.lr = 1e-3
        self.wd = 5e-4
        self.params_opt = 'hilbert'
        self.opt1 = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.current_loss = -1

        # MISC
        self.compute_grid()
        self.ims = []
        self.fig = []
        self.ims_hist = []
        self.fig_hist = []

        self.fig_hist, self.fig_axes = plt.subplots(2, 1)
        self.fig = plt.figure(0)

    def compute_grid(self):
        xrange = np.linspace(self.xmin, self.xmax, 70)
        yrange = np.linspace(self.ymin, self.ymax, 70)
        xx, yy = np.meshgrid(xrange, yrange)
        grid = np.array([xx.flatten(), yy.flatten()]).transpose()
        self.grid = torch.tensor(grid, dtype=torch.float32).to(self.device)
        self.xx = xx
        self.yy = yy

    def forward(self, x):
        x = self.model['hilbert'](x)
        if self.do_tanh: x = self.model['tanh'](3 * x)
        return x

    def evaluate(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tilde = self.forward(x)
        return y_tilde

    def loss(self, x, y):
        x_tilde = self.forward(x)
        c = .5 * self.criterion(x_tilde, y)

        if self.gamma is not 0:
            reg = .5 * self.model['hilbert'].regularization_term()
            return self.gamma * c + reg, c, reg
        else:
            return c, c, 0

    def custom_train(self, x, y, max_iter=50000, min_cost=0, plot=False):
        # LSSVM
        alpha, beta, loss, recon, reg = LSSVM.dual(x, y, self.sigma, self.gamma)
        print(alpha)
        print("Bias: %f" % beta)
        print("Reconstruction: %f" % recon)
        print("Regularization %f" % reg)

        # PUT MODEL IN MEMORY
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.special_init(x, y)
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
            y_tilde = self.forward(x)
            loss, recon, reg = self.loss(x, y)
            self.current_loss = loss.item()
            self.recon = recon.item()
            self.reg = reg.item()
            loss.backward(create_graph=True)
            return loss, y_tilde

        # TRAINING LOOP
        t = trange(max_iter, desc='ML')
        self.t_start = time.time()
        for epoch in t:
            self.opt.zero_grad()
            self.opt.step(closure)
            correct_alpha()

            t.set_description('ML (recon=%g, reg=%g)' % (self.recon, self.reg))

            if (epoch % 250) == 0:
                if self.do_plot: self.plot(x, y, alpha)
                if self.do_aggregate: self.aggregate()

        self.cpu()

        if self.do_plot:
            print('Generating GIF')

            ani = animation.ArtistAnimation(self.fig, self.ims)
            ani.save('imgs/Plot_2D.gif', writer='pillow')

            ani2 = animation.ArtistAnimation(self.fig_hist, self.ims_hist)
            ani2.save('imgs/Plot_hist.gif', writer='pillow')

    def plot(self, x, y, alpha):
        def plot_2d(x, y):
            sigma = self.model['hilbert'].model['expkernel'].sigma_trainable.data
            bias = self.model['hilbert'].model['lin'].bias.data
            alpha = self.model['hilbert'].model['lin'].weight
            alpha = alpha.detach().cpu().numpy().squeeze()
            p = self.model['hilbert'].model[self.kernel_type].param
            p = p.detach().cpu().numpy()
            p1 = p[0, :]
            p2 = p[1, :]

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            ## SURFACE PLOT
            surf = self.forward(self.grid)
            surf = surf.view(self.xx.shape).detach().cpu().numpy()
            cmap = plt.get_cmap('RdYlGn')

            # TITLE
            s1 = 'Number of support vectors: {:d}'.format(self.num_kernels)
            s2 = '$\sigma$ = {:f}'.format(sigma)
            s3 = 'Bias = {:f}'.format(bias)
            s = s1 + '\n' + s2 + '\n' + s3

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            # self.fig.clf()
            ax = self.fig.gca()
            self.fig.set_tight_layout(True)
            im1 = ax.contourf(self.xx, self.yy, surf, alpha=.4, cmap=cmap)
            im2 = ax.scatter(x[:, 0], x[:, 1], s=3, c=np.where(y == -1, 'k', 'b'))
            im3 = ax.scatter(p1, p2, s=100 * np.abs(alpha) / (np.max(np.abs(alpha))), c=np.where(alpha >= 0, 'g', 'r'))
            lim1 = ax.set_xlim((self.xmin, self.xmax))
            lim2 = ax.set_ylim((self.ymin, self.ymax))
            title = plt.text(0.5, 1.01, s, ha="center", va="bottom",
                             transform=ax.transAxes, fontsize="large")
            # plt.pause(0.01)
            # plt.show()

            im = [im1.collections[:], [im2, im3, title]]
            new_im = [y for x in im for y in x]
            self.ims.append(new_im)

        def plot_digits():
            alpha = self.model['hilbert'].model['lin'].weight
            alpha = alpha.detach().numpy().squeeze()

            p = self.model['hilbert'].model[self.kernel_type].param
            p = p.detach().numpy()

            num_row = 3
            num_col = 3
            num = num_row * num_col
            images = p[:num, :, :, :].squeeze()
            labels = alpha[:num].squeeze()

            fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
            for i in range(num):
                ax = axes[i // num_col, i % num_col]
                ax.imshow(images[i, :, :], cmap='gray')
                ax.set_title('Alpha: {:f}'.format(labels[i]))
            plt.tight_layout()
            plt.pause(0.1)
            plt.show()

        def plot_histogram(x, y, lssvm_alpha):
            alpha = self.model['hilbert'].model['lin'].weight
            alpha = alpha.detach().cpu().numpy().squeeze()

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            # self.fig.clf()
            self.fig_hist.set_tight_layout(True)
            self.fig_axes[0].set_xlim((-1, len(lssvm_alpha)))
            im1 = self.fig_axes[0].bar(range(len(lssvm_alpha)), np.abs(lssvm_alpha),
                                       color=np.where(lssvm_alpha >= 0, 'g', 'r'))
            self.fig_axes[1].set_ylim((0,np.max(np.abs(alpha))))
            self.fig_axes[1].set_xlim((-1,len(lssvm_alpha)))
            im2 = self.fig_axes[1].bar(range(len(alpha)), np.abs(alpha), color=np.where(alpha >= 0, 'g', 'r'))
            plt.pause(0.01)
            plt.show()

            im = [im1, im2]
            new_im = [y for x in im for y in x]
            self.ims_hist.append(new_im)

        if self.kernel_type == 'convkernel':
            plot_digits()
        else:
            plot_2d(x, y)
            plot_histogram(x, y, alpha)

    def special_init(self, x, y):
        # idx = torch.randperm(len(x))[:self.num_kernels]
        # x = x[idx,:]
        # y = y[idx]

        self.model['hilbert'].model[self.kernel_type]._special_init(x)
        self.model['hilbert'].model['lin']._special_init(y)

    def aggregate(self, tol_dist=1e-1, tol_alpha=5e-2):
        # MERGE
        def merge(tol_dist):
            p = self.model['hilbert'].model[self.kernel_type].param.detach().numpy().transpose()
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
            idx = np.argwhere(torch.abs(w) / max_w < tol_alpha)
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
