import kerch
import torch
import numpy as np
from matplotlib import pyplot as plt

# we define our new kernel
class MyImplicit(kerch.kernel.Implicit):
    def _implicit(self, x, y):
        # x: torch.Tensor of size [num_x, self.dim_input]
        # y: torch.Tensor of size [num_y, self.dim_input]

        k = torch.log(x @ y.T + 1)

        # return torch.Tensor of size [num_x, num_y]
        return k

# now we can just use the kernel
t = np.expand_dims(np.arange(0,15), axis=1)
sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

k = MyImplicit(sample=sample)

plt.figure()
plt.imshow(k.K)
plt.colorbar()
plt.title("Kernel Matrix")