# coding=utf-8
def reverse_dict(val: dict) -> dict:
    return {v: k for k, v in val.items()}