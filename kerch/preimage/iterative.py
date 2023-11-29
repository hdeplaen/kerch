import torch
from ..kernel import _Base as K
from .._cache import _Cache
from .smoother import smoother


def iterative(k_coeff: torch.Tensor, kernel: K, num_iter: int = 100) -> torch.Tensor:

    # DEFENSIVE
    if _Cache.cache_level_switcher[kernel.cache_level] > _Cache.cache_level_switcher['lightweight']:
        kernel._log.warn(f"The cache level is recommended to be at lightweight at maximum in order to ease the memory "
                         f"load during the pre-image computation.")
    assert k_coeff.shape[1] == kernel.num_idx, (f"Pre-image: the provided kernel coefficients do not correspond to "
                                                f"the number of sample datapoints.")

    # PRELIMINARIES
    vals0 = smoother(k_coeff, kernel.sample, kernel.num_idx)
    vals = torch.nn.Parameter(vals0, requires_grad=True)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.NAdam(vals, lr=1.e-3)

    # OPTIMIZE
    for idx in range(num_iter):
        k_current = kernel(vals)
        loss(k_current, k_coeff)
        optimizer.step()

    return vals.data
