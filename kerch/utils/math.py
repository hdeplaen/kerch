# coding=utf-8
import torch
from ..feature.logger import _GLOBAL_LOGGER


def eigs(A, k=None, B=None, psd=True, sym=True):
    r"""
    Eigenvalue decomposition. This method is a wrapper calling other methods depending on the context. In a kernel 
    context, most matrices are symmetric because kernels also are. Hence, they are Hermitian and a faster SVD can be 
    used. Alternatively, all the eigenvalues are not necessarily required and we may thus skip the computation of a 
    full eigendecomposition. The goal of this method is to always choose the most efficient method depending on the
    context.
    
    :param A: Matrix to be decomposed.
    :param k: Number of greatest eigenpairs requested. Defaults to `None`, which corresponds to computing all of them.
    :param B: Matrix in the case of a generalized eigenvalue problem. Specify `None` (default) for a classical
        eigenvalue decomposition.
    :param psd: Specifies whether the matrix `A` is positive semi-definite. Defaults to `True`.
    :param sym: Specifies whether the matrix `A` is positive symmetric. Defaults to `True`.
    :return: eigenvalues, eigenvectors.

    :type A: torch.Tensor
    :type k: int, optional
    :type B: torch.Tensor, optional
    :type psd: bool, optional
    :type sym: bool, optional
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    assert A is not None, 'Cannot decompose an empty matrix.'
    k1, k2 = A.shape
    assert k1 == k2, f'This function can only decompose square matrices (found {k1}x{k2}).'
    if k is None: k = k1
    assert k <= k1, f'Requested eigenvectors ({k}) exceeds matrix dimensions ({k1}).'

    try:
        s, v = torch.lobpcg(A, k=k, B=B, largest=True)
        _GLOBAL_LOGGER._logger.info('Using LOBPCG for eigendecomposition.')
    except:
        if sym:
            if B is None:
                s, v = torch.linalg.eigh(A)
            else:
                s, v = torch.linalg.eigh(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._logger.info('Using hermitian eigendecomposition (eigh).')
            v = v[:, -k:]  # eigenvectors are vertical components of v
            s = s[-k:]
            v = torch.flip(v, dims=(1,))
            s = torch.flip(s, dims=(0,))
        elif psd:
            if B is None:
                _, s, v = torch.svd(A)
            else:
                _, s, v = torch.svd(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._logger.info('Using SVD for eigendecomposition (svd).')
            v = v[:, :k]  # eigenvectors are vertical components of v
            s = s[:k]
        else:
            if B is None:
                s, v = torch.linalg.eig(A)
            else:
                s, v = torch.linalg.eig(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._logger.info('Using classical eigendecomposition (eig).')
            v = v[:, :k]  # eigenvectors are vertical components of v
            s = s[:k]

    return s.data, v.data
