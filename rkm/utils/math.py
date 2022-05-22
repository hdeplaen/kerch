import torch

def eigs(A, k=None, B=None, psd=True):
    assert A is not None, 'Cannot decompose an empty matrix.'
    k1, k2 = A.shape
    assert k1 == k2, 'This function can only decompose square matrices.'
    if k is None: k = k1
    assert k <= k1, 'Requested eigenvectors exceeds matrix dimensions.'

    try:
        s, v = torch.lobpcg(A, k=k, B=B, largest=True)
    except:
        if psd:
            if B is None:
                _, s, v = torch.svd(A)
            else:
                _, s, v = torch.svd(torch.linalg.inv(B) @ A)
        else:
            if B is None:
                _, s, v = torch.linalg.eig(A)
            else:
                _, s, v = torch.linalg.eig(torch.linalg.inv(B) @ A)
        v = v[:, 0:k]  # eigenvectors are vertical components of v
        s = s[:k]

    return s.data, v.data