# coding=utf-8
import matplotlib.pyplot as plt
import torch

from .elements import *
from ..module._Sample import _Sample
from ..kernel._BaseKernel import _BaseKernel
from ..level._View import _View
from ..level._KPCA import _KPCA
from ..utils import NotInitializedError


def current_sample(sample: _Sample, labels: bool = False) -> plt.Figure:
    assert isinstance(sample, _Sample), 'The sample argument does not contain any sample.'
    return plot_sample(sample.current_sample, labels=labels)


def K(kernel, labels: bool = False) -> plt.Figure:
    assert isinstance(kernel, _BaseKernel), 'kernel must be a _BaseKernel object'
    return plot_K(kernel.K, labels=labels)


def C(kernel, labels: bool = False) -> plt.Figure:
    assert isinstance(kernel, _BaseKernel), 'kernel must be a _BaseKernel object'
    return plot_C(kernel.C, labels=labels)


def Phi(kernel, labels: bool = False) -> plt.Figure:
    assert isinstance(kernel, _BaseKernel), 'kernel must be a _BaseKernel object'
    return plot_Phi(kernel.Phi, labels=labels)


def W(view, labels: bool = False) -> plt.Figure:
    assert isinstance(view, _View), 'The argument view does not contain (primal) weights.'
    return plot_W(view.W, labels=labels)


def hidden(view, labels: bool = False) -> plt.Figure:
    assert isinstance(view, _View), 'The argument view does not contain hidden variables.'
    return plot_hidden(view.H, labels=labels)


def hidden_correlation(view, labels: bool = False) -> plt.Figure:
    assert isinstance(view, _View), 'The argument view does not contain hidden variables.'
    return plot_hidden_correlation(view.H, labels=labels)


def forward(view, labels: bool = False) -> plt.Figure:
    assert isinstance(view, _View), 'The argument view cannot perform a forward pass.'
    return plot_forward(view(), labels=labels)


def _get_coord(val) -> torch.Tensor:
    val_min, val_max = torch.min(val), torch.max(val)
    val_diff = val_max - val_min
    val_min, val_max = val_min - 0.1 * val_diff, val_max + 0.1 * val_diff
    return torch.linspace(val_min, val_max, 50, dtype=val.dtype, device=val.device)


def scatter(view, dims=None, dim_target: int = 0, heat_map: bool = False, labels: bool = False) -> plt.Figure:
    assert isinstance(view, _View), 'The view argument is incorrect.'

    if dims is None:
        dims = [0, 1]

    sample = view.current_sample.data.detach()
    x = sample[:, dims[0]]
    y = sample[:, dims[1]]

    cmap = plt.get_cmap('RdYlGn')
    fig = plt.figure()
    ax = fig.gca()

    # specify colors if the target exists
    try:
        t = view.target.data[:, dim_target].detach().cpu()
        ax.scatter(x.data.detach().cpu(), y.data.detach().cpu(), s=5, c=cmap(255 * (t == 1.)))
    except NotImplementedError or AttributeError:
        ax.scatter(x.data.detach().cpu(), y.data.detach().cpu(), s=5)

    # labels
    if labels:
        ax.set_xlabel(f"dim_output[{dims[0]}]")
        ax.set_ylabel(f"dim_output[{dims[1]}]")
        ax.set_title(f"Forward")

    # heat map
    if heat_map:
        if view.dim_input != 2:
            view._log.warning('Cannot plot the heatmap for more than 2 input dimensions.')
            return fig
        x_coord = _get_coord(x)
        y_coord = _get_coord(y)
        x_grid, y_grid = torch.meshgrid(x_coord, y_coord)
        grid = torch.cat((x_grid.squeeze(), y_grid.squeeze()), dim=1)

        with torch.no_grad():
            eval_grid = view.forward(grid)[:, dim_target].data.detach().cpu()
        ax.contourf(x_grid.detach().cpu(), y_grid.detach().cpu(),
                    eval_grid.view(x_grid.shape), cmap=cmap, alpha=.5)

    return fig


def eigenvalues(kpca, labels: bool = False) -> plt.Figure:
    assert isinstance(kpca, _KPCA), 'The argument kpca does not contain eigenvalues.'
    return plot_eigenvalues(kpca.vals, labels=labels)
