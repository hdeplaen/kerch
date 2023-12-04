import torch
from ..kernel import _Base as K

def knn(k_coefficients: torch.Tensor, kernel: K, num: int = 1) -> torch.Tensor:
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

    num_points, num_coefficients = k_coefficients.shape[0]
    sample = K.current_sample
    num_sample = K.num_idx

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

    preimages = []
    for idx in range(num_points):
        _, indices = torch.sort(k_coefficients, descending=True)
        loc_sol = torch.mean(sample[indices[:num], :], dim=0)
        preimages.append(loc_sol)
    return torch.vstack(preimages)
