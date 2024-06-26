# coding=utf-8
from kerch.utils.type import EPS
from ..Transform import Transform
import torch

class UnitVarianceNormalization(Transform):
    def __init__(self, explicit: bool, default_path: bool = False, **kwargs):
        super(UnitVarianceNormalization, self).__init__(explicit=explicit,
                                                        name="Unit Variance Normalization", default_path=default_path, **kwargs)

    def _explicit_statistics(self, sample):
        return torch.std(sample, dim=0)

    def _explicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm, min=EPS)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) / torch.clamp(self.statistics_oos(x=x), min=EPS)

    def _revert_explicit(self, sample):
        return sample * torch.clamp(self.statistics_sample(), min=EPS)
