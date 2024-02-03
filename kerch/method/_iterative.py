import torch
from ..kernel._base_kernel import _BaseKernel as K
from ..feature.cache import Cache
from ._smoother import smoother
from ..utils import castf


def iterative(obj, x0: torch.Tensor, num_iter: int = 100, lr=1.e-3):
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \mathtt{obj}(\mathbf{x}).

    The method optimizes with an SGD algorithm.


    :param obj: Objective to minimize.
    :param x0: Starting value for the optimization.
    :type x0: torch.Tensor [num_points, dim_input]
    :param num_iter: Number of iterations for the optimization process. Defaults to 100.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Defaults to 0.001.
    :type lr: float, optional
    :return: Solution :math:`\tilde{\mathbf{x}}`
    :rtype: torch.Tensor [num_points, dim_input]
    """
    # PRELIMINARIES
    x0 = castf(x0)

    x = torch.nn.Parameter(x0, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)

    # OPTIMIZE
    def closure():
        optimizer.zero_grad()
        loss = obj(x)
        loss.backward(retain_graph=True)
        return loss

    for idx in range(num_iter):
        optimizer.step(closure)

    # RETURN
    return x.data


def iterative_preimage_k(k_image: torch.Tensor, kernel: K, num_iter: int = 100, lr=1.e-3,
                       light_cache=True) -> torch.Tensor:
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \big\lVert \mathtt{k\_image} - \mathtt{kernel.k(x)} \big\rVert_2^2

    The method optimizes with an SGD algorithm.

    :param k_image: coefficients in the RKHS to be inverted.
    :type k_image: torch.Tensor [num_points, num_idx]
    :param kernel: kernel on which this RKHS is based.
    :type kernel: :py:class:`kerch.kernel.Kernel` instance.
    :param num_iter: Number of iterations for the optimization process. Defaults to 100.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Defaults to 0.001.
    :type lr: float, optional
    :param light_cache: Specifies whether the cache has to made lighter during the pre-image to avoid keeping the
        statistics of each iteration. This results in a speedup. Defaults to ``True``.
    :type light_cache: bool, optional
    :return: Pre-image
    :rtype: torch.Tensor [num_points, dim_input]
    """
    k_image = castf(k_image)

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

    loss_fn = torch.nn.MSELoss(reduction='none')
    x0 = smoother(weights=k_image, observations=kernel.current_sample)

    def obj(vals):
        k_current = kernel.k(x=vals)
        return loss_fn(k_current, k_image)

    sol = iterative(obj=obj, x0=x0, num_iter=num_iter, lr=lr)

    # SET BACK THE ORIGINAL CACHE LEVEL
    if (cache_level > Cache._cache_level_switcher['light']) and light_cache:
        kernel.cache_level = cache_level

    return sol



def iterative_preimage_phi(phi_image: torch.Tensor, kernel: K, num_iter: int = 100, lr=1.e-3,
                       light_cache=True) -> torch.Tensor:
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \big\lVert \mathtt{phi\_image} - \mathtt{kernel.phi(x)} \big\rVert_2^2

    The method optimizes with an SGD algorithm.

    :param phi_image: feature map image to be inverted.
    :type phi_image: torch.Tensor [num_points, dim_feature]
    :param kernel: kernel on which this RKHS is based.
    :type kernel: :py:class:`kerch.kernel.Kernel` instance.
    :param num_iter: Number of iterations for the optimization process. Defaults to 100.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Defaults to 0.001.
    :type lr: float, optional
    :param light_cache: Specifies whether the cache has to made lighter during the pre-image to avoid keeping the
        statistics of each iteration. This results in a speedup. Defaults to ``True``.
    :type light_cache: bool, optional
    :return: Pre-image
    :rtype: torch.Tensor [num_points, dim_feature]
    """
    phi_image = castf(phi_image)

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
        (f"Pre-image: the provided explicit feature map images dimensions ({phi_image.size(1)}) do not correspond to the "
         f"feature dimension of the provided kernel ({kernel.dim_feature}).")

    assert num_iter > 0, \
        f"The number of iterations num_iter ({num_iter}) must be strictly positive (num_iter > 0)."

    loss_fn = torch.nn.MSELoss(reduction='none')
    weights = phi_image @ kernel.Phi.T
    x0 = smoother(weights=weights, observations=kernel.current_sample)

    def obj(vals):
        phi_current = kernel.phi(x=vals)
        return loss_fn(phi_current, phi_image)

    sol = iterative(obj=obj, x0=x0, num_iter=num_iter, lr=lr)

    # SET BACK THE ORIGINAL CACHE LEVEL
    if (cache_level > Cache._cache_level_switcher['light']) and light_cache:
        kernel.cache_level = cache_level

    return sol
