import torch
from ..kernel import _Base as K
from .._cache import _Cache
from .smoother import smoother


def iterative(k_coefficient: torch.Tensor, kernel: K, num_iter: int = 100, lr=1.e-3, lightweight_cache=True) -> torch.Tensor:
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \big\lVert \mathtt{k_coefficient} - \mathtt{kernel.k(x)} \big\rVert_2^2


    The method optimizes with an NAdam algorithm.

    :param k_coefficients: coefficients in the RKHS to be inverted.
    :type k_coefficients: torch.Tensor
    :param kernel: kernel on which this RKHS is based.
    :type kernel: Kernel class from kerch.
    :param num_iter: Number of iterations for the optimization process. Defaults to 100.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Default to 0.001.
    :type lr: float, optional
    :param lightweight_cache: Specifies whether the cache has to made lighter during the pre-image to avoid keeping the
        statistics of each iteration. This results in a speedup. Defaults to True.
    :type lightweight_cache: bool, optional
    :return: Pre-image
    :rtype: torch.Tensor
    """

    # CHECK IF THE CACHE LEVEL HAS TO BE CHANGED
    cache_level = _Cache.cache_level_switcher[kernel.cache_level]
    if cache_level > _Cache.cache_level_switcher['lightweight']:
        if lightweight_cache:
            kernel.cache_level = 'lightweight'
        else:
            kernel._log.warn(f"The cache level is recommended to be at lightweight at maximum in order to ease the "
                             f"memory load during the pre-image computation. It is temporarily being set to "
                             f"lightweight. You can also set the argument lightweight_cache to True and set it "
                             f"temporarily lower during the computation.")

    assert k_coefficient.shape[1] == kernel.num_idx, (f"Pre-image: the provided kernel coefficients do not correspond to "
                                                f"the number of sample datapoints.")

    # PRELIMINARIES
    vals0 = smoother(k_coefficient, kernel, 'all')
    vals = torch.nn.Parameter(vals0, requires_grad=True)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.NAdam(vals, lr=lr)

    # OPTIMIZE
    for idx in range(num_iter):
        k_current = kernel(vals)
        loss(k_current, k_coefficient)
        optimizer.step()

    # SET BACK THE ORIGINAL CACHE LEVEL
    if (cache_level > _Cache.cache_level_switcher['lightweight']) and lightweight_cache:
        kernel.cache_level = cache_level

    # RETURN
    return vals.data
