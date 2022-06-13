import torch
from .._logger import _GLOBAL_LOGGER


def eigs(A, k=None, B=None, psd=True):
    assert A is not None, 'Cannot decompose an empty matrix.'
    k1, k2 = A.shape
    assert k1 == k2, 'This function can only decompose square matrices.'
    if k is None: k = k1
    assert k <= k1, 'Requested eigenvectors exceeds matrix dimensions.'

    try:
        s, v = torch.lobpcg(A, k=k, B=B, largest=True)
        _GLOBAL_LOGGER._log.info('Using LOBPCG for eigendecomposition.')
    except:
        if psd:
            if B is None:
                _, s, v = torch.svd(A)
            else:
                _, s, v = torch.svd(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._log.info('Using SVD for eigendecomposition.')
        else:
            if B is None:
                s, v = torch.linalg.eig(A)
            else:
                s, v = torch.linalg.eig(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._log.info('Using classical (EIG) eigendecomposition.')
        v = v[:, 0:k]  # eigenvectors are vertical components of v
        s = s[:k]

    return s.data, v.data
