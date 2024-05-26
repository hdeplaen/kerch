# coding=utf-8
def reverse_dict(val: dict) -> dict:
    r"""
    Inverts to keywords and the values in a dictionnary.

    :param val: Dictionary to be inverted.
    :return: Inverted dictionnary.

    :type val: dict
    :rtype: dict
    """
    return {v: k for k, v in val.items()}