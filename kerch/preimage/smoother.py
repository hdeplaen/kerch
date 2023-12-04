import torch
from ..kernel import _Base as K
from typing import Union


def smoother(k_coefficients: torch.Tensor, kernel: K, num: Union[int, str] = 'all') -> torch.Tensor:
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
    num_points, num_coefficients = k_coefficients.shape
    sample = K.current_sample
    num_sample = K.num_idx

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

    if num == num_coefficients:
        return torch.einsum('ni,ij->nj', k_coefficients / torch.sum(k_coefficients, dim=0), sample)

    preimages = []
    for idx in range(num_points):
        sorted_coefficients, indices = torch.sort(k_coefficients[idx, :], descending=True)
        nearest_coefficients = sorted_coefficients[:num]

        normalized_coefficients = nearest_coefficients / torch.sum(nearest_coefficients)
        loc_sol = torch.einsum('i,ij->j', normalized_coefficients, sample[indices[:num], :])
        preimages.append(loc_sol)
    return torch.vstack(preimages)
