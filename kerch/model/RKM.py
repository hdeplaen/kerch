# coding=utf-8
from __future__ import annotations
import nested_dict

from .Model import Model
from ..utils import capitalize_only_first, KerchError


class RKM(Model):
    def __init__(self, *args, **kwargs):
        super(RKM, self).__init__(*args, **kwargs)
        level = 0
        while True:
            try:
                level_kwargs = kwargs[f"level{level}"]
                self.append_level(**level_kwargs)
                level += 1
            except KeyError:
                break
        if level == 0:
           raise KerchError(cls=self, message='The RKM model does not contain any level. '
                                              'Please add one or check if the arguments are correct.')

    def __str__(self):
        description = f"[Model] RKM with the following levels:"
        nl, tab = '\n', '\t'
        for num, level in enumerate(self.levels):
            dim_input = level.dim_input
            dim_output = level.dim_output
            description += f"{nl}{tab}* L{num}[{dim_input},{dim_output}]: " \
                           f"{capitalize_only_first(level.representation)} {level}"
        return description

    def append_level(self, *args, **kwargs) -> None:
        self._append_level(*args, **kwargs)




