def invert_dict(name, dict):
    # The name becomes the key inside the key and the original key become the names.
    # The output is of the form [(new_name1, new_dict1), (new_name2, new_dict2), ...].
    dicts = []
    for key in dict:
        new_dict = {name: dict[key]}
        dicts.append((key, new_dict))
    return dicts