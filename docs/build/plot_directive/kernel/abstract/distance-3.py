import kerch
import torch
import numpy as np
from matplotlib import pyplot as plt

# we define our l1 kernel
class MyL1Distance(kerch.kernel.Distance):
    def __init__(self, *args, **kwargs):
        super(kerch.kernel.Distance, self).__init__(*args, **kwargs)

    def _dist(self, x, y):
        # x: torch.Tensor of size [num_x, dim]
        # y: torch.Tensor of size [num_y, dim]
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y

        # return torch.Tensor of size [num_x, num_y]
        return torch.sum(torch.abs(diff), dim=0, keepdim=False)


# we define out kernel that is going to use the distances
class MyKernel(kerch.kernel.SelectDistance):
    def __init__(self, *args, **kwargs):
        super(kerch.kernel.SelectDistance, self).__init__(*args, **kwargs)

    def _implicit(self, x, y):
        # x: torch.Tensor of size [num_x, dim]
        # y: torch.Tensor of size [num_y, dim]
        # return torch.Tensor of size [num_x, num_y]
        return -self._dist(x, y)

# we define our sample
t = np.expand_dims(np.arange(0,15), axis=1)
sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

# we define our two kernels, the first one with our defined distance, the second one with a pre-defined distance
k1 = MyKernel(sample=sample, distance=MyL1Distance)
k2 = MyKernel(sample=sample, distance='euclidean')

# plot
plt.figure(1)
plt.imshow(k1.K)
plt.colorbar()
plt.title("L1")

plt.figure(2)
plt.imshow(k2.K)
plt.colorbar()
plt.title("L2")