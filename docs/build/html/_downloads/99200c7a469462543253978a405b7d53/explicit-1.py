import kerch
import torch
import numpy as np
from matplotlib import pyplot as plt

# we define our new kernel
class MyExplicit(kerch.kernel.Explicit):
    def _explicit(self, x):
        # x: torch.Tensor of size [num, self.dim_input]
        phi1 = x                                                    # [num, self.dim_input]
        phi2 = x ** 2                                               # [num, self.dim_input]
        phi3 = torch.log(torch.sum(x * x, dim=1, keepdim=True)+1)   # [num, 1]

        phi = torch.cat((phi1, phi2, phi3), dim=1)                  # [num, 2*self.dim_input + 1]

        # return torch.Tensor of size [num, self.dim_feature]
        # if not specified (see further), self.dim_feature will be determined automatically
        return phi

# now we can just use the kernel
t = np.expand_dims(np.arange(0,15), axis=1)
sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

k = MyExplicit(sample=sample)


fig, axs = plt.subplots(1,2)
axs[0].imshow(sample)
axs[0].set_title("Sample")
im = axs[1].imshow(k.Phi)
axs[1].set_title("Feature Map")
fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')


plt.figure()
plt.imshow(k.K)
plt.colorbar()
plt.title("Kernel Matrix")