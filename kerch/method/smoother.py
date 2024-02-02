# coding=utf-8
from __future__ import annotations

import torch
from typing import Union

from ..utils import DEFAULT_KERNEL_TYPE
from kerch.utils import castf


@torch.no_grad()
def smoother(coefficients: torch.Tensor, observations: torch.Tensor,
             num: Union[int, str] = 'all') -> torch.Tensor:
    r"""
    Returns a weighted sum of the observations by the coefficients.

    .. math::
        \mathtt{out}_{i,j} = \frac{\sum_l\mathtt{weights}_{i,l} * \mathtt{observations}_{l,j}}{\sum_l\mathtt{weights}_{i,l}}.

    :param coefficients: coefficients used in the smoother.
    :type coefficients: torch.Tensor [num_points, num_observations]
    :param observations: observation corresponding to each weight dimension.
    :type observations: torch.Tensor [num_observations, dim_observations]
    :param num: Number of closest points to be used. Either an integer representing the number or
        the string ``'all'``. Defaults to ``'all'``.
    :type num: int or str, optional
    :return: Weighted observations
    :rtype: torch.Tensor [num_points, dim_observations]
    """

    # PRELIMINARIES
    coefficients = castf(coefficients)
    observations = castf(observations)

    num_points, num_coefficients = coefficients.shape
    num_observations = observations.shape[0]

    # DEFENSIVE
    assert num_coefficients == num_observations, \
        f'Smoother: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points observations points({num_observations}).'

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
        f"The argument num ({num}) exceeds the number of observations ({num_observations})."

    # PRE-IMAGE
    if num == num_coefficients:
        kept_coeff = coefficients
    else:
        vals, indices = torch.topk(coefficients, k=num, dim=1, largest=True)
        kept_coeff = torch.zeros_like(coefficients)
        kept_coeff.scatter_(1, indices, vals)
    normalized_coeff = kept_coeff / torch.sum(kept_coeff, dim=1, keepdim=True)
    return normalized_coeff @ observations



@torch.no_grad()
def kernel_smoother(domain: torch.Tensor, observations: torch.Tensor, num: Union[int, str] = 'all', kernel_type: str=DEFAULT_KERNEL_TYPE, **kwargs) -> torch.Tensor:
    r"""
    Smoothens the function :math:`f(\mathtt{domain}_{i,:}) = \mathtt{observation}_{i,:}` with weights defined
    as a kernel on the domain.

    .. math::
        \mathtt{out}_{i,j} = \frac{\sum_l k\left(\mathtt{domain}_{i,:},\mathtt{domain}_{l,:}\right) * \mathtt{observations}_{l,j}}{\sum_l k\left(\mathtt{domain}_{i,:},\mathtt{domain}_{l,:}\right)}.

    The kernel is defined as in :py:func:`kerch.kernel.factory`.

    :param domain: domain corresponding to each observation.
    :type domain: torch.Tensor [num_observations, dim_domain]
    :param observations: observation corresponding to each domain entry.
    :type observations: torch.Tensor [num_observations, dim_observations]
    :param num: Number of closest points to be used. Either an integer representing the number or
        the string ``'all'``. Defaults to ``'all'``.
    :type num: int or str, optional
    :param kernel_type: Type of kernel chosen. For the possible choices, please refer to the `Factory Type` column of the
        :doc:`../kernel/index` documentation. Defaults to :py:data:`kerch.DEFAULT_KERNEL_TYPE`.
    :param \**kwargs: Arguments to be passed to the kernel constructor, such as `sample` or `sigma`. If an argument is
        passed that does not exist (e.g. `sigma` to a `linear` kernel), it will just be neglected. For the default
        values, please refer to the default values of the requested kernel.
    :type kernel_type: str, optional
    :type \**kwargs: dict, optional
    :return: Smoothed function :math:`f` according to kernel :math:`k`.
    :rtype: torch.Tensor [num_observations, dim_observations]
    """
    domain = castf(domain)
    observations = castf(domain)

    assert domain.shape[0] == observations.shape[0], f"Not the same number of domain {domain.shape[0]} and coefficients points {domain.shape[0]}."

    from ..kernel import factory

    k = factory(kernel_type=kernel_type, sample=domain, **kwargs)
    return smoother(coefficients=k.K, observations=observations, num=num)