from __future__ import annotations

from ._Model import _Model
from kerch.train.Trainer import Trainer


def create_from_yaml(filename: str) -> tuple[_Model | None, Trainer | None]:
    return rkm_from_yaml(filename), trainer_from_yaml(filename)


def rkm_from_yaml(filename: str) -> _Model | None:
    return None


def trainer_from_yaml(filename: str) -> Trainer | None:
    return None
