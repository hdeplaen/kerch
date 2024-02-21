from typing import Iterator

from ..explicit import Explicit
from .random_features import RandomFeatures
from .._factory import factory

class RFStacked(Explicit):
    def __init__(self, *args, **kwargs):
        self._kernels = list()
        super(RFStacked, self).__init__(*args, **kwargs)

        self._rf_kernel_type = kwargs.pop('rf_kernel_type', 'rf_lrelu')
        self._num_weights = kwargs.pop('num_weights', [100])
        self._inner_sample_transform = kwargs.pop('inner_sample_transform', ['standardize'])
        assert isinstance(self._num_weights, (list, tuple)), 'num_weights must be a list or tuple'
        assert all(isinstance(n, int) for n in self._num_weights), 'num_weights must be a list of integers'
        for nw in self.num_weights:
            self._append(nw)

    def __str__(self):
        val = 'stacked random features:'
        for k in self.kernels:
            val += f'\n\t{k}'
        return val

    def _append(self, num_weights: int):
        kwargs = {'kernel_type': self._rf_kernel_type,
                  'num_weights': num_weights,
                  'sample_transform': self._inner_sample_transform,}
        if self.num_kernels == 0:
            kwargs['sample'] = self.current_sample_projected
        else:
            kwargs['sample'] = self._kernels[-1].phi()
        kernel = factory(**kwargs)
        assert isinstance(kernel, RandomFeatures), 'rf_kernel_type must be an instance of RandomFeatures'
        self._kernels.append(kernel)

    def init_sample(self, sample=None, idx_sample=None, prop_sample=None):
        super(RFStacked, self).init_sample(sample, idx_sample, prop_sample)
        s = self.current_sample_projected
        for k in self.kernels:
            k.init_sample(s)
            s = k.phi()

    @property
    def kernels(self) -> Iterator[RandomFeatures]:
        yield from self._kernels

    def kernel(self, num: int = 0) -> RandomFeatures:
        return self._kernels[num]


    @property
    def num_kernels(self) -> int:
        return len(self._kernels)

    @property
    def num_weights(self) -> list[int]:
        return self._num_weights

    @property
    def hparams_fixed(self):
        return {"Kernel": "Random Features Stacked", 
                "Number of Weights": self.num_weights,
                "RF Type": self._rf_kernel_type,
                "Inner Sample Transform": self._inner_sample_transform,
                **super(RFStacked, self).hparams_fixed}

    def _explicit(self, x):
        for k in self.kernels:
            x = k.phi(x)
        return x

    def _explicit_preimage(self, phi):
        for k in reversed(self._kernels):
            phi = k.explicit_preimage(phi, method='explicit')
        return phi

