import torch
from kerch.kernel import _BaseKernel as K
from kerch._module._Cache import _Cache
from .smoother import smoother


def iterative(k_coefficient: torch.Tensor, kernel: K, num_iter: int = 100, lr=1.e-3, light_cache=True) -> torch.Tensor:
    r"""
    Minimizes to following problem for each point in order to find the preimage:

    .. math::
        \tilde{\mathbf{x}} = \mathrm{argmin}_{\mathbf{x}} \big\lVert \mathtt{k_coefficient} - \mathtt{kernel.k(x)} \big\rVert_2^2


    The method optimizes with an SGD algorithm.

    :param k_coefficients: coefficients in the RKHS to be inverted.
    :type k_coefficients: torch.Tensor
    :param kernel: kernel on which this RKHS is based.
    :type kernel: Kernel class from kerch.
    :param num_iter: Number of iterations for the optimization process. Defaults to 100.
    :type num_iter: int, optional
    :param lr: Learning rate of the optimizer. Default to 0.001.
    :type lr: float, optional
    :param light_cache: Specifies whether the cache has to made lighter during the pre-image to avoid keeping the
        statistics of each iteration. This results in a speedup. Defaults to True.
    :type light_cache: bool, optional
    :return: Pre-image
    :rtype: torch.Tensor
    """

    # CHECK IF THE CACHE LEVEL HAS TO BE CHANGED
    cache_level = _Cache._cache_level_switcher[kernel.cache_level]
    if cache_level > _Cache._cache_level_switcher['light']:
        if light_cache:
            kernel.cache_level = 'light'
        else:
            kernel._log.warn("The cache level is recommended to be at light at maximum in order to ease the "
                             "memory load during the pre-image computation. It is temporarily being set to "
                             "light. You can also set the argument light_cache to True to set it "
                             "temporarily lower during the pre-image computation.")

    assert k_coefficient.size(1) == kernel.num_idx, \
        f"Pre-image: the provided kernel coefficients ({k_coefficient.size(1)}) do not correspond to the number " \
        f"of sample datapoints ({kernel.num_idx})."

    assert num_iter > 0, \
        f"The number of iterations num_iter ({num_iter}) must be strictly positive (num_iter > 0)."

    # PRELIMINARIES
    vals0 = smoother(k_coefficient, kernel, 'all')
    vals = torch.nn.Parameter(vals0, requires_grad=True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([vals], lr=lr)

    # OPTIMIZE
    def closure():
        optimizer.zero_grad()
        k_current = kernel(vals)
        loss = loss_fn(k_current, k_coefficient)
        loss.backward(retain_graph=True)
        return loss

    for idx in range(num_iter):
        optimizer.step(closure)

    # SET BACK THE ORIGINAL CACHE LEVEL
    if (cache_level > _Cache._cache_level_switcher['light']) and light_cache:
        kernel.cache_level = cache_level

    # RETURN
    return vals.data
