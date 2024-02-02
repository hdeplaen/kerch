# coding=utf-8
import torch
from ..utils import castf, DEFAULT_KERNEL_TYPE


@torch.no_grad()
def knn(dists: torch.Tensor, observations: torch.Tensor, num: int = 1) -> torch.Tensor:
    r"""
    For each distance ``dists``, returns the average of the ``num`` smallest corresponding observations.

    :param dists: coefficients used in the knn.
    :type dists: torch.Tensor [num_points, num_observations]
    :param observations: observation corresponding to each weight dimension.
    :type observations: torch.Tensor [num_observations, dim_observations]
    :param num: number of nearest neighbors. Defaults to 1.
    :type num: int, optional
    :return: KNN
    :rtype: torch.Tensor [num_points, dim_observations]
    """

    # PRELIMINARIES
    dists = castf(dists)
    observations = castf(observations)

    num_points, num_coefficients = dists.shape
    num_observations = observations.shape[0]

    # DEFENSIVE
    try:
        num = int(num)
    except ValueError:
        raise ValueError('The argument num is not an integer.')

    assert num_coefficients == num_observations, \
        f'KNN: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points ({num_observations}).'

    assert num <= num_coefficients, \
        (f"Too much required neighbors ({num}) compared to the number of observations points "
         f"({num_observations}). Please insure that the argument num is not greater than the number of observations "
         f"points.")

    assert num > 0, \
        f"The number of required neighbors num must be strictly positive ({num})."

    # PRE-IMAGE
    _, indices = torch.topk(-dists, k=num, dim=1)
    kept_sample = observations[indices]
    return torch.mean(kept_sample, dim=1)


@torch.no_grad()
def kernel_knn(domain: torch.Tensor, observations: torch.Tensor, num: int = 1, kernel_type: str = DEFAULT_KERNEL_TYPE,
               **kwargs) -> torch.Tensor:
    r"""
    For each coefficient, returns the average of the ``num`` greatest corresponding kernel values on the domain.
    The kernel is defined as in :py:func:`kerch.kernel.factory`.

    :param domain: domain corresponding to each observation.
    :type domain: torch.Tensor [num_observations, dim_domain]
    :param observations: observation corresponding to each domain entry.
    :type observations: torch.Tensor [num_observations, dim_observations]
    :param num: number of nearest neighbors. Defaults to 1.
    :type num: int, optional
    :param kernel_type: Type of kernel chosen. For the possible choices, please refer to the `Factory Type` column of the
        :doc:`../kernel/index` documentation. Defaults to :py:data:`kerch.DEFAULT_KERNEL_TYPE`.
    :param \**kwargs: Arguments to be passed to the kernel constructor, such as `sample` or `sigma`. If an argument is
        passed that does not exist (e.g. `sigma` to a `linear` kernel), it will just be neglected. For the default
        values, please refer to the default values of the requested kernel.
    :type kernel_type: str, optional
    :type \**kwargs: dict, optional
    :return: KNN
    :rtype: torch.Tensor [num_points, dim_observations]
    """
    domain = castf(domain)
    observations = castf(domain)

    assert domain.shape[0] == observations.shape[
        0], f"Not the same number of domain {domain.shape[0]} and coefficients points {domain.shape[0]}."

    from ..kernel import factory

    k = factory(kernel_type=kernel_type, sample=domain, **kwargs)
    return knn(dists=-k.K, observations=observations, num=num)
