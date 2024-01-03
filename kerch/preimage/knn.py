import torch
from ..kernel import _Base

@torch.no_grad()
def knn(k_coefficients: torch.Tensor, kernel: _Base, num: int = 1) -> torch.Tensor:
    r"""
    Returns the sum of the num closest x, the distance being given by the coefficients.

    :param k_coefficients: coefficients in the RKHS to be inverted.
    :type k_coefficients: torch.Tensor
    :param kernel: kernel on which this RKHS is based.
    :type kernel: Kernel class from kerch.
    :param num: number of nearest neighbors. Defaults to 1.
    :type num: int, optional
    :return: Pre-image
    :rtype: torch.Tensor
    """

    # PRELIMINARIES
    num_points, num_coefficients = k_coefficients.shape
    sample = kernel.current_sample_projected
    num_sample = kernel.num_idx

    # DEFENSIVE
    try:
        num = int(num)
    except ValueError:
        raise ValueError('The argument num is not an integer.')

    assert num_coefficients == num_sample, \
        f'KNN: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points ({num_sample}).'

    assert num <= num_coefficients, \
        (f"Too much required neighbors ({num}) compared to the number of sample points in the provided kernel "
         f"({num_sample}). Please insure that the argument num is not greater than the number of sample points.")

    assert num > 0, \
        f"The number of required neighbors num must be strictly positive ({num})."

    # PRE-IMAGE
    _, indices = torch.topk(k_coefficients, k=num, dim=1, largest=True)
    kept_sample = sample[indices]
    preimages = torch.mean(kept_sample, dim=1)

    return preimages
