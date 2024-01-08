import torch
from kerch.kernel import _BaseKernel
from typing import Union

@torch.no_grad()
def smoother(k_coefficients: torch.Tensor, kernel: _BaseKernel, num: Union[int, str] = 'all') -> torch.Tensor:
    r"""
    Returns a weighted sum of x by the coefficients.

    :param k_coefficients: coefficients in the RKHS to be inverted.
    :type k_coefficients: torch.Tensor
    :param kernel: kernel on which this RKHS is based.
    :type kernel: Kernel class from kerch.
    :param num: Number of closest points to be used. Either an integer representing the number or
        the string 'all'. Defaults to 'all'.
    :type num: int or str, optional
    :return: Pre-image
    :rtype: torch.Tensor
    """

    # PRELIMINARIES
    num_points, num_coefficients = k_coefficients.shape
    sample = kernel.current_sample_projected
    num_sample = kernel.num_idx

    # DEFENSIVE
    assert num_coefficients == num_sample, \
        f'Smoother: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points sample points in the provided kernel ({num_sample}).'

    if isinstance(num, str):
        assert num.lower() == 'all', \
            f"Only the string value of 'all' is allowed as value for the argument num ({num})." \
            f"Please use an integer is you prefer specifying a specific number."
        num = num_coefficients
    else:
        try:
            num = int(num)
        except ValueError:
            raise ValueError('The argument num is not an integer.')

    assert num > 0, \
        f"The argument num ({num}) must be a strictly positive integer (num > 0)."

    assert num <= num_coefficients, \
        f"The argument num ({num}) exceeds the number of sample point of the provided kernel ({num_sample})."

    # PRE-IMAGE
    if num == num_coefficients:
        kept_coeff = k_coefficients
    else:
        vals, indices = torch.topk(k_coefficients, k=num, dim=1, largest=True)
        kept_coeff = torch.zeros_like(k_coefficients)
        kept_coeff.scatter_(1, indices, vals)
    normalized_coeff = kept_coeff / torch.sum(kept_coeff, dim=1, keepdim=True)
    preimages = normalized_coeff @ sample

    return preimages
