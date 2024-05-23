import numpy as np

from ..model.Model import Model
from .._logger import _GLOBAL_LOGGER
from ..rkm._level import _Level
from matplotlib import pyplot as plt

def classifier_level(lvl: _Level, dims=None):
    if dims is None:
        dims = [0, 1]

    # get the sample
    try:
        sample = lvl.sample.detach().cpu().numpy()
        x = sample[:,dims[0]]
        y = sample[:,dims[1]]
        t = lvl.targets.detach().cpu().numpy()
    except AttributeError:
        _GLOBAL_LOGGER._log.warning("Could not plot based on the training points as these seem not "
                                    "to exist.")
        return

    # create grid
    def _get_coord(val):
        val_min, val_max = np.min(val), np.max(val)
        val_diff = val_max - val_min
        val_min, val_max = val_min - 0.1 * val_diff, val_max + 0.1 * val_diff
        return np.linspace(val_min, val_max, 50)

    x_coord = _get_coord(x)
    y_coord = _get_coord(y)
    x_grid, y_grid = np.meshgrid(x_coord, y_coord)
    grid = np.concatenate((np.expand_dims(x_grid.flatten(), axis=1),
                           np.expand_dims(y_grid.flatten(), axis=1)),
                          axis=1)

    # evaluate
    eval_grid = lvl.forward(grid).detach().cpu()

    # PLOTS
    cmap = plt.get_cmap('RdYlGn')
    plt.figure()
    plt.scatter(x, y, s=5, c=cmap(255*(t==1.)))
    plt.contourf(x_grid, y_grid, eval_grid.view(x_grid.shape), cmap=cmap, alpha=.5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(lvl.__class__.__name__)
    plt.show()

def model(mdl:Model, dims=None):
    if dims is None:
        dims = [0, 1]

    # get the sample
    try:
        sample = mdl._training_data.cpu().numpy()
        x = sample[:,dims[0]]
        y = sample[:,dims[1]]
        t = mdl._training_labels
    except AttributeError:
        _GLOBAL_LOGGER._log.warning("Could not plot based on the training points as these seem not "
                                    "to exist.")
        return

    # create grid
    def _get_coord(val):
        val_min, val_max = np.min(val), np.max(val)
        val_diff = val_max - val_min
        val_min, val_max = val_min - 0.1 * val_diff, val_max + 0.1 * val_diff
        return np.linspace(val_min, val_max, 50)

    x_coord = _get_coord(x)
    y_coord = _get_coord(y)
    x_grid, y_grid = np.meshgrid(x_coord, y_coord)
    grid = np.concatenate((np.expand_dims(x_grid.flatten(), axis=1),
                           np.expand_dims(y_grid.flatten(), axis=1)),
                          axis=1)

    # evaluate
    eval_grid = mdl.forward(grid)

    # PLOTS
    cmap = plt.get_cmap('RdYlGn')
    plt.figure()
    plt.scatter(x, y, s=5, c=cmap(255*(t==1.)))
    plt.contourf(x_grid, y_grid, eval_grid.view(x_grid.shape), cmap=cmap, alpha=.5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(mdl.__class__.__name__)
    plt.show()
