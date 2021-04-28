from arch.kernel_utils import *
import torch.nn as nn
from tqdm import trange
from sklearn import metrics
from arch.hessianfree import (HessianFree)
from arch.hyperplane_optimizer import *
import time

# PLOTS
import matplotlib.pyplot as plt
import numpy as np

class Hilbert(nn.Module):
    def __init__(self, size_in, num_kernels, size_out, sigma):
        super(Hilbert, self).__init__()
        self.size_in = size_in
        self.num_kernels = num_kernels
        self.size_out = size_out
        self.sigma = sigma

        self.model = nn.ModuleDict({
            'convkernel': ConvKernel(self.num_kernels),
            'expkernel': ExponentialKernel(self.size_in, self.num_kernels, self.sigma),
            'lin': CustomLinear(self.num_kernels, self.size_out)
        })

    def forward(self, x):
        x = self.model['convkernel'](x)
        x = self.model['lin'](x)
        return x

    def special_init(self, x, y):
        self.model['conkernel']._special_init(x)
        self.model['lin']._special_init(y)

class KernelNetwork(nn.Module):
    def __init__(self, num_data, num_kernels, num_classes, sigma=1, cuda=True, range=(-1,1,-1,1)):
        super(KernelNetwork, self).__init__()

        if torch.cuda.is_available() and cuda:
            self.device = torch.device('cuda')
            torch.cuda.empty_cache()
            print('Using CUDA')
        else:
            self.device = torch.device('cpu')

        # PARAMETERS
        self.num_data = num_data
        self.num_kernels = num_kernels
        self.num_classes = num_classes
        self.sigma = sigma
        self.xmin = range[0]
        self.xmax = range[1]
        self.ymin = range[2]
        self.ymax = range[3]

        # MODEL
        self.criterion = nn.MSELoss()
        self.model = nn.ModuleDict({
            'hilbert': Hilbert(self.num_data, self.num_kernels, self.num_classes, self.sigma),
            'tanh': nn.Tanh()
        })

        # TRAINING ALGORITHMS
        self.lr = 1e-3
        self.wd = 5e-4
        self.params_opt = 'hilbert'
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.opt2 = HessianFree(self.model.parameters(), use_gnm=True, verbose=False)
        self.current_loss = -1

        # MISC
        self.compute_grid()

    def compute_grid(self):
        xrange = np.linspace(self.xmin,self.xmax, 70)
        yrange = np.linspace(self.ymin,self.ymax, 70)
        xx, yy = np.meshgrid(xrange,yrange)
        grid = np.array([xx.flatten(), yy.flatten()]).transpose()
        self.grid = torch.tensor(grid , dtype=torch.float32).to(self.device)
        self.xx = xx
        self.yy = yy

    def forward(self, x):
        x = self.model['hilbert'](x)
        x = self.model['tanh'](3*x)
        return x

    def loss(self, x, y):
        x_tilde = self.forward(x)
        return self.criterion(x_tilde, y)

    def custom_train(self, x, y, max_iter=25000, min_cost=0, plot=False):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.special_init(x,y)
        self.to(self.device)

        def correct_alpha():
            alpha = self.model['hilbert'].model['lin'].weight.data
            alpha -= torch.mean(alpha)
            self.model['hilbert'].model['lin'].weight.data = alpha

        def closure():
            self.opt.zero_grad()
            y_tilde = self.model['hilbert'](x)
            loss = self.loss(x, y)
            self.current_loss = loss.item()
            loss.backward(create_graph=True)
            return loss, y_tilde

        old_loss = 100

        t = trange(max_iter, desc='ML')
        self.t_start = time.time()
        for epoch in t:
            self.opt.zero_grad()
            self.opt.step(closure)
            correct_alpha()

            # print_mean()
            # train optimal support vector values through more steps, or LSSVM

            loss_item = self.current_loss
            t.set_description('ML (loss=%g)' % loss_item)

            # if np.abs(old_loss-loss_item) < 1e-9:
            #     break
            #     # print(old_loss/loss_item)

            if plot and ((epoch % 500)) == 0:
                old_loss = loss_item
                # self.plot_2d(x,y)
                self.plot_digits()
                # self.aggregate()

            # if (epoch % 2000) == 0:
            #     correct_alpha()

        self.cpu()

    def plot_2d(self, x, y):
        alpha = self.model['hilbert'].model['lin'].weight
        alpha = alpha.detach().numpy().squeeze()
        p = self.model['hilbert'].model['convkernel'].param
        p = p.detach().numpy()
        p1 = p[0, :]
        p2 = p[1, :]

        x = x.detach().numpy()
        y = y.detach().numpy()

        surf = self.forward(self.grid)
        surf = surf.view(self.xx.shape).detach().numpy()
        cmap = plt.get_cmap('RdYlGn')

        s1 = 'Number of support vectors: {:d}'.format(self.num_kernels)
        s2 = 'Sum of support vector values: {:f}'.format(np.sum(alpha))
        s = s1 + '\n' + s2

        plt.figure(0)
        plt.clf()
        plt.contourf(self.xx, self.yy, surf, alpha=.4, cmap=cmap)
        plt.scatter(x[:,0], x[:,1], s=3, c=np.where(y==-1,'k','b'))
        plt.scatter(p1, p2, s=100*np.abs(alpha)/(np.max(np.abs(alpha))), c=np.where(alpha>=0, 'g', 'r'))
        plt.xlim((self.xmin, self.xmax))
        plt.ylim((self.ymin, self.ymax))
        plt.title(s)
        plt.pause(0.1)
        plt.show()

    def plot_digits(self):
        alpha = self.model['hilbert'].model['lin'].weight
        alpha = alpha.detach().numpy().squeeze()

        p = self.model['hilbert'].model['convkernel'].param
        p = p.detach().numpy()

        num_row = 3
        num_col = 3
        num = num_row*num_col
        images = p[:num,:,:,:].squeeze()
        labels = alpha[:num].squeeze()

        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i in range(num):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(images[i,:,:], cmap='gray')
            ax.set_title('Alpha: {:f}'.format(labels[i]))
        plt.tight_layout()
        plt.pause(0.1)
        plt.show()


    def special_init(self, x, y):
        idx = torch.randperm(len(x))[:self.num_kernels]
        x = x[idx,:]
        y = y[idx]

        self.model['hilbert'].model['convkernel']._special_init(x)
        self.model['hilbert'].model['lin']._special_init(y)

    def aggregate(self, tol_dist=2e-1, tol_alpha=5e-5):
        # MERGE
        def merge(tol_dist):
            p = self.model['hilbert'].model['convkernel'].param.detach().numpy().transpose()
            D = np.triu(metrics.pairwise_distances(p),k=1)
            mask = np.logical_and(D<tol_dist, D>0)
            idx = np.argwhere(mask)
            num_fusions = idx.shape[0]

            def merge_points(idx1, idx2):
                self.num_kernels -= 1
                self.model['hilbert'].model['convkernel'].merge_kernels(idx1,idx2)
                self.model['hilbert'].model['lin'].merge_kernels(idx1, idx2)

            for i in range(num_fusions):
                idx1 = idx[i,0]
                idx2 = idx[i,1]
                idx[idx == idx2] = idx1
                idx[idx > idx2] -= 1
                merge_points(idx1, idx2)

        # SUPPRESS ALPHA
        def suppress(tol_alpha):
            w = self.model['hilbert'].model['lin'].weight.data
            idx = np.argwhere(w<tol_alpha)
            num_suppressions = len(idx)

            def suppress_point(idx_loc):
                self.num_kernels -= 1
                self.model['hilbert'].model['convkernel'].suppress_kernel(idx_loc)
                self.model['hilbert'].model['lin'].suppress_kernel(idx_loc)

            for i in range(num_suppressions):
                idx_loc = idx[0,i]
                suppress_point(idx_loc)
                idx[idx > idx_loc] -= 1

        merge(tol_dist)
        # suppress(tol_alpha)

        # UPDATE OPTIMIZERS
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.opt2 = HessianFree(self.model.parameters(), use_gnm=True, verbose=False)
        self.model['hilbert'].zero_grad()
