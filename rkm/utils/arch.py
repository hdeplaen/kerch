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


def verify_dim(M):
    return M


def process_y(y):
    if len(list(y.shape)) == 1:
        y = y.unsqueeze(1)
    return y
