from __future__ import annotations

from ._Model import _Model
from ..utils import capitalize_only_first


class RKM(_Model):
    def __init__(self, *args, **kwargs):
        super(RKM, self).__init__(*args, **kwargs)

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




