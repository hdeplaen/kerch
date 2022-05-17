import torch


def invert_dict(name, dict):
    # The name becomes the key inside the key and the original key become the names.
    # The output is of the form [(new_name1, new_dict1), (new_name2, new_dict2), ...].
    dicts = []
    for key in dict:
        new_dict = {name: dict[key]}
        dicts.append((key, new_dict))
    return dicts


def add_dict(name, dict):
    dicts = {}
    for key in dict:
        new_entry = {name + ' ' + key: dict[key].data}
        dicts = {**new_entry, **dicts}
    return dicts


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


def verify_dim(M):
    return M


def process_y(y):
    if len(list(y.shape)) == 1:
        y = y.unsqueeze(1)
    return y
