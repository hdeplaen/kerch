import kerch
import torch
import numpy as np
from matplotlib import pyplot as plt

# we define our l1 kernel
class MyExponential(kerch.kernel.Exponential):
    def _dist(self, x, y):
        # x: torch.Tensor of size [num_x, dim]
        # y: torch.Tensor of size [num_y, dim]
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y

        # return torch.Tensor of size [num_x, num_y]
        return torch.sum(torch.abs(diff), dim=0, keepdim=False)

# we define our sample
t = np.expand_dims(np.arange(0,15), axis=1)
sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

# now we can just use the kernel
k = MyExponential(sample=sample)

plt.figure(1)
plt.imshow(k.K)
plt.colorbar()
plt.title("Sigma = "+str(k.sigma))

k.sigma = 1

plt.figure(2)
plt.imshow(k.K)
plt.colorbar()
plt.title("Sigma = "+str(k.sigma))