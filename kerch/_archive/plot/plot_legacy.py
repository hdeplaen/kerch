from kerch import kwargs_decorator

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class plot():
    @kwargs_decorator({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1,
                           "cuda": False, "live": False,
                           "x": [], "y": [], "lssvm_alpha": [],
                           "kernel_type": 'expkernel'})
    def __init__(self, model, **kwargs):
        # CUDA
        self.cuda = kwargs['cuda']
        if torch.cuda.is_available() and self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.cuda = False

        # MODEL & PLOTS
        self.mdl = model
        self.live = kwargs["live"]

        self.fig_two_dim = plt.figure(0)
        self.fig_digits = plt.figure(1)
        self.fig_histogram, self.ax_histogram = plt.subplots(2,1)

        self.ims_two_dim = []
        self.ims_digits = []
        self.ims_histogram = []

        self.xmin = kwargs["xmin"]
        self.xmax = kwargs["xmax"]
        self.ymin = kwargs["ymin"]
        self.ymax = kwargs["ymax"]

        self.x = kwargs["x"]
        self.y = kwargs["y"]
        self.lssvm_alpha = kwargs["lssvm_alpha"]
        self.kernel_type = kwargs["kernel_type"]

        self._compute_grid()

    def _compute_grid(self):
        xrange = np.linspace(self.xmin, self.xmax, 70)
        yrange = np.linspace(self.ymin, self.ymax, 70)
        xx, yy = np.meshgrid(xrange, yrange)
        grid = np.array([xx.flatten(), yy.flatten()]).transpose()
        self.grid = torch.tensor(grid, dtype=torch.float32).to(self.device)
        self.xx = xx
        self.yy = yy

    def _two_dim(self):
        sigma = self.mdl.Model['hilbert'].Model['expkernel'].sigma_trainable.data
        bias = self.mdl.Model['hilbert'].Model['lin'].bias.data
        alpha = self.mdl.Model['hilbert'].Model['lin'].weight
        alpha = alpha.detach().cpu().numpy().squeeze()
        p = self.mdl.Model['hilbert'].Model[self.mdl.kernel_type].param
        p = p.detach().cpu().numpy()
        p1 = p[0, :]
        p2 = p[1, :]

        ## SURFACE PLOT
        surf = self.mdl.forward(self.grid)
        surf = surf.View(self.xx.shape).detach().cpu().numpy()
        cmap = plt.get_cmap('RdYlGn')

        # TITLE
        s1 = 'Number of support vectors: {:d}'.format(self.mdl.num_sample)
        s2 = '$\sigma$ = {:f}'.format(sigma)
        s3 = 'Bias = {:f}'.format(bias)
        s = s1 + '\n' + s2 + '\n' + s3

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if self.live: self.fig_two_dim.clf()
        ax = self.fig_two_dim.gca()
        self.fig_two_dim.set_tight_layout(True)
        im1 = ax.contourf(self.xx, self.yy, surf, alpha=.4, cmap=cmap)
        im2 = ax.scatter(self.x[:, 0], self.x[:, 1], s=3, c=np.where(self.y == -1, 'k', 'b'))
        im3 = ax.scatter(p1, p2, s=100 * np.abs(alpha) / (np.max(np.abs(alpha))), c=np.where(alpha >= 0, 'g', 'r'))
        lim1 = ax.set_xlim((self.xmin, self.xmax))
        lim2 = ax.set_ylim((self.ymin, self.ymax))
        title = plt.text(0.5, 1.01, s, ha="center", va="bottom",
                         transform=ax.transAxes, fontsize="large")
        if self.live:
            plt.pause(0.01)
            plt.show()
        else:
            im = [im1.collections[:], [im2, im3, title]]
            new_im = [y for x in im for y in x]
            self.ims_two_dim.append(new_im)

    def _digits(self):
        alpha = self.mdl.Model['hilbert'].Model['lin'].weight
        alpha = alpha.detach().numpy().squeeze()

        p = self.mdl.Model['hilbert'].Model["converkel"].param
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
        # plt.pause(0.1)
        # plt.show()

    def _histogram(self):
        alpha = self.mdl.Model['hilbert'].Model['lin'].weight
        alpha = alpha.detach().cpu().numpy().squeeze()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if self.live:
            self.fig_histogram.clf()

        # self.fig_histogram.set_tight_layout(True)
        self.ax_histogram[0].set_xlim((-1, len(self.lssvm_alpha)))
        im1 = self.ax_histogram[0].bar(range(len(self.lssvm_alpha)), np.abs(self.lssvm_alpha),
                                   color=np.where(self.lssvm_alpha >= 0, 'g', 'r'))
        self.ax_histogram[1].set_ylim((0, np.max(np.abs(alpha))))
        self.ax_histogram[1].set_xlim((-1, len(self.lssvm_alpha)))
        im2 = self.ax_histogram[1].bar(range(len(alpha)), np.abs(alpha), color=np.where(alpha >= 0, 'g', 'r'))
        if self.live:
            plt.pause(0.01)
            plt.show()
        else:
            im = [im1, im2]
            new_im = [y for x in im for y in x]
            self.ims_histogram.append(new_im)

    def imgen(self):
        if self.kernel_type == 'convkenel':
            self._digits()
        else:
            self._two_dim()
            self._histogram()

    def gifgen(self):
        def next_path(path_pattern):
            i = 1

            # First do an exponential search
            while os.path.exists(path_pattern % i):
                i = i * 2

            # Result lies somewhere in the interval (i/2..i]
            # We call this interval (a..b] and narrow it down until a + 1 = b
            a, b = (i // 2, i)
            while a + 1 < b:
                c = (a + b) // 2  # interval midpoint
                a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

            return b

        if self.live:
            print("Nothing to generate")
        else:
            print('Generating GIF')
            pathnumber = next_path('imgs/plot-%s-twodim.gif')

            # 2D
            ani_two_dim = animation.ArtistAnimation(self.fig_two_dim, self.ims_two_dim)
            ani_two_dim.save('imgs/plot-' + str(pathnumber) + '-twodim.gif', writer='pillow')

            # HISTOGRAM
            ani_histogram = animation.ArtistAnimation(self.fig_histogram, self.ims_histogram)
            ani_histogram.save('imgs/plot-' + str(pathnumber) + '-histogram.gif', writer='pillow')