import torch
from ._projection import _Projection
from kerch.utils.type import EPS

class _UnitSphereNormalization(_Projection):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_UnitSphereNormalization, self).__init__(explicit=explicit, name="Unit Sphere Normalization",
                                                       default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.norm(sample, dim=1, keepdim=True)

    def _implicit_statistics(self, sample, x=None):
        if sample.nelement() == 0:
            return self._implicit_self(x)
        else:
            return torch.sqrt(torch.diag(sample))[:, None]

    def _explicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm, min=EPS)

    def _implicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm * norm.T, min=EPS)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return torch.norm(oos, dim=1, keepdim=True)

    def _implicit_statistics_oos(self, x=None, oos=None) -> torch.Tensor:
        if oos.nelement() == 0:
            d = self.parent._implicit_diag(x)
        else:
            d = torch.diag(oos)[:, None]
        return torch.sqrt(d)

    def _explicit_oos(self, x=None):
        oos = self.parent.oos(x)
        norm = self.statistics_oos(x=x, oos=oos)
        return oos / torch.clamp(norm, min=EPS)

    def _implicit_oos(self, x=None, y=None):
        oos = self.parent.oos(x=x, y=y)
        # avoid computing the full matrix and use the _parent_diag when possible
        norm_x, norm_y = self.statistics_oos(x=x, y=y, oos=torch.empty(0))
        return oos / torch.clamp(norm_x * norm_y.T, min=EPS)