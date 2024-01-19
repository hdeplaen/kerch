# coding=utf-8

from __future__ import annotations
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor as T


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


def plot_eigenvalues(eigenvalues: T, labels: bool = False, log: bool = True, ax: plt.Axes | None = None) -> plt.Figure:
    r"""
    Bar plot of the eigenvalues.

    :param eigenvalues: Eigenvalues
    :param labels: If true, the title and axis labels are displayed. Defaults to False.
    :param log: If true, the magnitude is in logarithmic scale. En error will be thrown if some values are too close
        to zero or negative. Defaults to True.
    :param ax: plt.Axes instance. If None, a new figure/axis pair will be created. Defaults to None.
    :return: The figure

    :type eigenvalues: torch.Tensor[:]
    :type labels: bool, optional
    :type log: bool, optional
    :type ax: matplotlib.axes.Axes, optional
    :rtype: matplotlib.pyplot.Figure
    """
    fig, ax = _get_fig_ax(ax)
    eigenvalues = eigenvalues.squeeze()
    assert len(eigenvalues.shape) == 1, 'The vals argument must be a vector: each eigenvalue must be a scalar.'

    ax.bar(len(eigenvalues), eigenvalues.detach().cpu())

    if labels:
        ax.set_title('Eigenvalues')
        ax.set_xlabel('dim_output')
        ax.set_ylabel('Magnitude')

    if log:
        ax.set_yscale('log')

    return fig
