# coding=utf-8
from ..Transform import Transform
import torch

class MeanCentering(Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(MeanCentering, self).__init__(explicit=explicit, name="Mean centering", default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.mean(sample, dim=0)

    def _implicit_statistics(self, sample, x=None):
        mean = torch.mean(sample, dim=1, keepdim=True)
        mean_tot = torch.mean(mean)
        return mean, mean_tot

    def _explicit_sample(self):
        sample = self.parent.sample
        return sample - self.statistics_sample(sample)

    def _implicit_sample(self):
        mat = self.parent.sample
        mean, mean_tot = self.statistics_sample(mat)
        return mat - mean - mean.T + mean_tot

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _implicit_statistics_oos(self, x=None, oos=None):
        sample_x = self.parent.oos(x=x)
        return torch.mean(sample_x, dim=1, keepdim=True), self.statistics_sample()[1]

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) - self.statistics_oos(x=x)

    def _implicit_oos(self, x=None, y=None):
        mean_x, mean_y = self.statistics_oos(x=x, y=y)
        mean_tot = mean_x[1]
        return self.parent.oos(x=x, y=y) - mean_x[0] \
               - mean_y[0].T \
               + mean_tot

    def _revert_explicit(self, oos):
        return oos + self.statistics_sample()

