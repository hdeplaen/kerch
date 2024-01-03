import yaml

from .RKM import RKM
from .Trainer import Trainer


def create_from_yaml(filename: str) -> tuple[RKM | None, Trainer | None]:
    return rkm_from_yaml(filename), trainer_from_yaml(filename)


def rkm_from_yaml(filename: str) -> RKM | None:
    return None


def trainer_from_yaml(filename: str) -> Trainer | None:
    return None
