import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..kernel._base_kernel import _BaseKernel as K
from ..feature.cache import Cache
from ._knn import knn
from ..utils import castf
from tqdm import tqdm


def iterative(obj, x0: torch.Tensor, num_iter: int = 50, lr=1.e-3, verbose: bool = False):
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \mathtt{obj}(\mathbf{x}).

    The method optimizes with an SGD algorithm.

    :param verbose: Shows the training loop. Defaults to ``False``.
    :type verbose: bool, optional
    :param obj: Objective to minimize.
    :param x0: Starting value for the optimization.
    :type x0: torch.Tensor [num_points, dim_input]
    :param num_iter: Number of iterations for the optimization process. Defaults to 50.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Defaults to 0.001.
    :type lr: float, optional
    :return: Solution :math:`\tilde{\mathbf{x}}`
    :rtype: torch.Tensor [num_points, dim_input]
    """
    # PRELIMINARIES
    x0 = castf(x0)

    x = Variable(x0, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.8, patience=50, cooldown=50)

    # OPTIMIZE
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = obj(x)
        loss.backward(retain_graph=False)
        return loss

    if verbose:
        epochs = tqdm(range(num_iter))
    else:
        epochs = range(num_iter)
    for _ in epochs:
        l = optimizer.step(closure)
        scheduler.step(l)
        last_lr = scheduler._last_lr[0]
        if verbose:
            epochs.set_description(f"Loss: {l:1.2e}, lr: {last_lr:1.1e}")
        if last_lr < 1.e-5:
            break

    # RETURN
    return x.data


def iterative_preimage_k(k_image: torch.Tensor, kernel: K, num_iter: int = 50, lr=1.e-3,
                         light_cache=True, verbose: bool = False) -> torch.Tensor:
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \big\lVert \mathtt{k\_image} - \mathtt{kernel.k(x)} \big\rVert_2^2

    The method optimizes with an SGD algorithm.

    :param verbose: Shows the training loop. Defaults to ``False``.
    :type verbose: bool, optional
    :param k_image: coefficients in the RKHS to be inverted.
    :type k_image: torch.Tensor [num_points, num_idx]
    :param kernel: kernel on which this RKHS is based.
    :type kernel: :py:class:`kerch.kernel.Kernel` instance.
    :param num_iter: Number of iterations for the optimization process. Defaults to 50.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Defaults to 0.001.
    :type lr: float, optional
    :param light_cache: Specifies whether the cache has to made lighter during the pre-image to avoid keeping the
        statistics of each iteration. This results in a speedup. Defaults to ``True``.
    :type light_cache: bool, optional
    :return: Pre-image
    :rtype: torch.Tensor [num_points, dim_input]
    """
    k_image = castf(k_image).data

    # CHECK IF THE CACHE LEVEL HAS TO BE CHANGED
    cache_level = Cache._cache_level_switcher[kernel.cache_level]
    if cache_level > Cache._cache_level_switcher['light']:
        if light_cache:
            kernel.cache_level = 'light'
        else:
            kernel._logger.warning("The cache level is recommended to be at light at maximum in order to ease the "
                                   "memory load during the pre-image computation. It is temporarily being set to "
                                   "light. You can also set the argument light_cache to True to set it "
                                   "temporarily lower during the pre-image computation.")

    assert k_image.size(1) == kernel.num_idx, \
        f"Pre-image: the provided kernel coefficients ({k_image.size(1)}) do not correspond to the number " \
        f"of sample datapoints ({kernel.num_idx})."

    assert num_iter > 0, \
        f"The number of iterations num_iter ({num_iter}) must be strictly positive (num_iter > 0)."

    loss_fn = torch.nn.MSELoss()
    # x0 = torch.zeros(k_image.shape[0], kernel.dim_input, dtype=k_image.dtype)
    x0 = knn(dists=-k_image + k_image.max(), observations=kernel.current_sample)

    def obj(vals):
        k_current = kernel.k(x=vals)
        return loss_fn(k_current, k_image)

    sol = iterative(obj=obj, x0=x0, num_iter=num_iter, lr=lr, verbose=verbose)

    # SET BACK THE ORIGINAL CACHE LEVEL
    if (cache_level > Cache._cache_level_switcher['light']) and light_cache:
        kernel.cache_level = cache_level

    return sol


def iterative_preimage_phi(phi_image: torch.Tensor, kernel: K, num_iter: int = 50, lr=1.e-3,
                           light_cache=True, verbose: bool = False) -> torch.Tensor:
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \big\lVert \mathtt{phi\_image} - \mathtt{kernel.phi(x)} \big\rVert_2^2

    The method optimizes with an SGD algorithm.

    :param verbose: Shows the training loop. Defaults to ``False``.
    :type verbose: bool, optional
    :param phi_image: feature map image to be inverted.
    :type phi_image: torch.Tensor [num_points, dim_feature]
    :param kernel: kernel on which this RKHS is based.
    :type kernel: :py:class:`kerch.kernel.Kernel` instance.
    :param num_iter: Number of iterations for the optimization process. Defaults to 50.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Defaults to 0.001.
    :type lr: float, optional
    :param light_cache: Specifies whether the cache has to made lighter during the pre-image to avoid keeping the
        statistics of each iteration. This results in a speedup. Defaults to ``True``.
    :type light_cache: bool, optional
    :return: Pre-image
    :rtype: torch.Tensor [num_points, dim_feature]
    """
    phi_image = castf(phi_image).data

    # CHECK IF THE CACHE LEVEL HAS TO BE CHANGED
    cache_level = Cache._cache_level_switcher[kernel.cache_level]
    if cache_level > Cache._cache_level_switcher['light']:
        if light_cache:
            kernel.cache_level = 'light'
        else:
            kernel._logger.warning("The cache level is recommended to be at light at maximum in order to ease the "
                                   "memory load during the pre-image computation. It is temporarily being set to "
                                   "light. You can also set the argument light_cache to True to set it "
                                   "temporarily lower during the pre-image computation.")

    assert phi_image.size(1) == kernel.num_idx, \
        f"Pre-image: the provided explicit feature map images dimensions ({phi_image.size(1)}) do not correspond to " \
        f"the feature dimension of the provided kernel ({kernel.dim_feature})."

    assert num_iter > 0, \
        f"The number of iterations num_iter ({num_iter}) must be strictly positive (num_iter > 0)."

    loss_fn = torch.nn.MSELoss()
    weights = phi_image @ kernel.Phi.T
    x0 = knn(dists=-weights, observations=kernel.current_sample)

    def obj(vals):
        phi_current = kernel.phi(x=vals)
        return loss_fn(phi_current, phi_image)

    sol = iterative(obj=obj, x0=x0, num_iter=num_iter, lr=lr, verbose=verbose)

    # SET BACK THE ORIGINAL CACHE LEVEL
    if (cache_level > Cache._cache_level_switcher['light']) and light_cache:
        kernel.cache_level = cache_level

    return sol
