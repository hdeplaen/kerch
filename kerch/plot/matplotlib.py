# coding=utf-8

from __future__ import annotations
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor as T

from kerch.level import KPCA

mpl.rcParams['hatch.linewidth'] = 1.5
newline = '\n'


def _get_fig_ax(ax: plt.Axes | None = None) -> [plt.Figure, plt.Axes]:
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        assert isinstance(ax, plt.Axes), 'The argument ax is not an instance of matplotlib.Axes.'
        fig = ax.get_figure()
    return fig, ax


def _plot_matrix(m: T,
                 title: str | None = None,
                 x_label: str | None = None,
                 y_label: str | None = None,
                 labels: bool = True,
                 ax: plt.Axes | None = None) -> plt.Figure:
    fig, ax = _get_fig_ax(ax)
    im = ax.imshow(m.data.detach().cpu())
    ax.set_xticks([])
    ax.set_yticks([])

    # title and labels
    if title is not None and labels:
        ax.set_title(title)
    if x_label is None and labels:
        ax.set_xlabel(x_label)
    if y_label is None and labels:
        ax.set_ylabel(y_label)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    return fig


def plot_K(K: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots the kernel matrix.

    :param K: Kernel matrix
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type K: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(K,
                        title='Kernel Matrix',
                        x_label='num_sample',
                        y_label='num_sample',
                        labels=labels,
                        ax=ax)


def plot_K_reconstructed(K_reconstructed: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
        Plots the reconstructed kernel matrix.

        :param K_reconstructed: The reconstructed kernel matrix.
        :param labels: If true, the title and axis labels are displayed. Defaults to False.
        :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
        :return: The figure

        :type K_reconstructed: torch.Tensor[:,:]
        :type labels: bool, optional
        :type ax: matplotlib.axes.Axes, optional
        :rtype: matplotlib.pyplot.Figure
        """
    return _plot_matrix(K_reconstructed,
                        title="Reconstructed Kernel Matrix",
                        x_label='num_sample',
                        y_label='num_sample',
                        labels=labels,
                        ax=ax)


def plot_hidden(hidden: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots the hidden variables.

    :param hidden: Hidden variables
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type hidden: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(hidden,
                        title="Hidden Variables",
                        x_label='num_sample',
                        y_label='dim_output',
                        labels=labels,
                        ax=ax)


def plot_hidden_correlation(hidden: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots the inner product of the hidden variables.

    :param hidden: Hidden variables
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type hidden: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(hidden.T @ hidden,
                        title="Correlation of the Hidden Variables",
                        x_label='num_sample',
                        y_label='num_sample',
                        labels=labels,
                        ax=ax)


def plot_C(C: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots the inner product of the feature vectors.

    :param C: Inner product of the feature vectors.
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type C: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(C,
                        title="Correlation",
                        x_label='dim_feature',
                        y_label='dim_feature',
                        labels=labels,
                        ax=ax)


def plot_Phi(Phi: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots a feature vectors matrix.

    :param Phi: Feature vectors
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type Phi: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(Phi,
                        title="Feature Map",
                        x_label='num_sample',
                        y_label='dim_feature',
                        labels=labels,
                        ax=ax)

def plot_W(W: T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots a weight matrix.

    :param W: Weight matrix
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type W: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(W, title="Weights", x_label='dim_feature', y_label='dim_output', labels=labels, ax=ax)

def plot_sample(sample:T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots a sample.

    :param sample: Sample
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type sample: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(sample, title="Sample", x_label='num_sample', y_label='dim_intput', labels=labels, ax=ax)

def plot_forward(forward:T, labels: bool = False, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Plots the forward pass results.

    :param forward: Matrix containing the forward pass results.
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type forward: torch.Tensor[:,:]
    :type labels: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    return _plot_matrix(forward, title="Forward", x_label='num_sample', y_label='dim_output', labels=labels, ax=ax)


def plot_eigenvalues(kpca: KPCA,
                     labels: bool = False,
                     num_vals: int | None = None,
                     section_div: int | None = None,
                     log: bool = False,
                     ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Bar plot of the eigenvalues.

    :param section_div: divide the bar plot in two sections and show corresponding variance
    :param num_vals: number of eigenvalues to be plotted
    :param kpca: KPCA model
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param log: If true, the magnitude is in logarithmic scale. En error will be thrown if some values are too close
        to zero or negative. Defaults to True.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type kpca: kerch.level.KPCA
    :type labels: bool, optional
    :type log: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    fig, ax = _get_fig_ax(ax)
    vals = 100 * kpca.vals / kpca.total_variance(normalize=False)

    if num_vals is not None:
        vals = vals[:num_vals]
    num_vals = vals.shape[0]

    ax.bar(range(num_vals), vals, facecolor='none', edgecolor='k', hatch="///", linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(-.7, num_vals - 0.3)

    if section_div is not None:
        x = section_div - .5
        ax.axvline(x=x, color='k', linestyle='dashed', linewidth=2)
        ax.annotate(f"{vals[:section_div].sum():1.2f}%", xy=(0, 85))
        ax.annotate(f"{100 * kpca.relative_variance() - vals[:section_div].sum():1.2f}%", xy=(section_div + 1, 85))
        ax.fill([x, x, num_vals - .3, num_vals - .3], [0, 100, 100, 0], facecolor='k', edgecolor='none', alpha=.2)

    if labels:
        ax.set_ylabel("Explained variance [%]")
        ax.set_title(f"Total explained variance: {100 * kpca.relative_variance():3.2f}%")
        ax.set_xticks(range(num_vals), [f"$\lambda_{i + 1}$" for i in range(num_vals)])

    if log:
        ax.set_yscale('log')

    return fig


def _plot_individual(ax: plt.Axes, vals: T, title: str = ""):
    import math
    import torch

    vals = vals.squeeze()
    assert len(vals.shape) == 1, 'Incorrect shape to be plotted.'
    mean = torch.mean(vals)
    std = torch.std(vals)

    # hist
    ax.hist(vals, bins=15, density=True, alpha=0.5, color='k')

    # pdf
    fact = 1 / (std * math.sqrt(2 * math.pi))
    xmin, xmax = ax.get_xlim()
    x = torch.linspace(xmin, xmax, 100)
    y = fact * torch.exp(-.5 * (x - mean) ** 2 / (std ** 2))
    ax.plot(x, y, 'k', linewidth=2)

    # labels
    ax.set_title(title + newline + f"($\mu$={mean:1.2f}, $\sigma$={std:1.2f})")
    # ax.set_xlabel("Value")
    # ax.set_ylabel("PDF")


def plot_vals(vals: T, num_vals: int | None = None, title: str = ""):
    import math
    vals = vals.squeeze()
    if len(vals.shape) == 1:
        vals = vals[:, None]
    assert len(vals.shape) == 2, 'Incorrect shape, must be of dimension 2.'

    if num_vals is not None:
        vals = vals[:, :num_vals]

    num_plots = vals.shape[1]
    num_columns = math.ceil(num_plots ** (.5))
    num_rows = math.ceil(num_plots / num_columns)
    fig, axs = plt.subplots(num_rows, num_columns)

    for i, ax in enumerate(axs.ravel()):
        try:
            _plot_individual(ax, vals[:, i], f"Component {i + 1}")
        except IndexError:
            break
    fig.suptitle(title)
    fig.tight_layout(pad=.5)
    return fig
