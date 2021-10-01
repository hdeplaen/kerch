import torch

def invert_dict(name, dict):
    # The name becomes the key inside the key and the original key become the names.
    # The output is of the form [(new_name1, new_dict1), (new_name2, new_dict2), ...].
    dicts = []
    for key in dict:
        new_dict = {name: dict[key]}
        dicts.append((key, new_dict))
    return dicts

def eigs(A, k=None):
    assert A is not None, 'Cannot decompose empty matrix.'
    k1, k2 = A.shape
    assert k1==k2, 'This function can only decompose square matrices.'
    if k is None: k=k1
    assert k <= k1, 'Requested eigenvectors exceeds matrix dimensions.'

    try:
        s, v = torch.lobpcg(A, k=k, largest=True)
    except:
        _, s, v = torch.svd(A)
        v = v[:, 0:k]  # eigenvectors are vertical components of v
        s = s[:k]

    return s, v

def verify_dim(M):
    return M
