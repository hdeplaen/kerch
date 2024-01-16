import torch
from ..module._Logger import _GLOBAL_LOGGER


def eigs(A, k=None, B=None, psd=True, sym=True):
    assert A is not None, 'Cannot decompose an empty matrix.'
    k1, k2 = A.shape
    assert k1 == k2, f'This function can only decompose square matrices (found {k1}x{k2}).'
    if k is None: k = k1
    assert k <= k1, f'Requested eigenvectors ({k}) exceeds matrix dimensions ({k1}).'

    try:
        s, v = torch.lobpcg(A, k=k, B=B, largest=True)
        _GLOBAL_LOGGER._log.info('Using LOBPCG for eigendecomposition.')
    except:
        if sym:
            if B is None:
                s, v = torch.linalg.eigh(A)
            else:
                s, v = torch.linalg.eigh(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._log.info('Using hermitian eigendecomposition (eigh).')
            v = v[:, -k:]  # eigenvectors are vertical components of v
            s = s[-k:]
        elif psd:
            if B is None:
                _, s, v = torch.svd(A)
            else:
                _, s, v = torch.svd(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._log.info('Using SVD for eigendecomposition (svd).')
            v = v[:, :k]  # eigenvectors are vertical components of v
            s = s[:k]
        else:
            if B is None:
                s, v = torch.linalg.eig(A)
            else:
                s, v = torch.linalg.eig(torch.linalg.inv(B) @ A)
            _GLOBAL_LOGGER._log.info('Using classical eigendecomposition (eig).')
            v = v[:, :k]  # eigenvectors are vertical components of v
            s = s[:k]

    return s.data, v.data
