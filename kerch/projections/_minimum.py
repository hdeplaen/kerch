from ._projection import _Projection
import torch

class _MinimumCentering(_Projection):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MinimumCentering, self).__init__(explicit=explicit,
                                                name="Minimum Centering", default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.min(sample, dim=0).values

    def _explicit_sample(self):
        sample = self.parent.sample
        return sample - self.statistics_sample(sample)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) - self.statistics_oos(x=x)

    def _revert_explicit(self, oos):
        return oos + self.statistics_sample()
