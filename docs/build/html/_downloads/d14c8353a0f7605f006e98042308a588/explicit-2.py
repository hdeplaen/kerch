import kerch
import torch
import numpy as np
from matplotlib import pyplot as plt

# we define our new kernel
class MyExplicit(kerch.kernel.Explicit):
    def _explicit(self, x):
        # x: torch.Tensor of size [num, dim]
        phi1 = x                                                    # [num, dim]
        phi2 = x ** 2                                               # [num, dim]
        phi3 = torch.log(torch.sum(x * x, dim=1, keepdim=True)+1)   # [num, 1]

        phi = torch.cat((phi1, phi2, phi3), dim=1)                  # [num, 2*dim + 1]

        # return torch.Tensor of size [num, dim_feature]
        # if not specified (see further), dim_feature will be determined automatically
        return phi

# we define the sample
t = np.expand_dims(np.arange(0,15), axis=1)
sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

# we also define an out_of_sample
t_oos = np.expand_dims(np.arange(15,30), axis=1)
oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

# we initialize our new kernel
k = MyExplicit(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['standard'])

# sample
fig, axs = plt.subplots(1,3)
axs[0].imshow(sample)
axs[0].set_title("Original")
axs[1].imshow(k.current_sample_projected)
axs[1].set_title("Transformed")
im = axs[2].imshow(k.Phi)
axs[2].set_title("Feature Map")
fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
fig.suptitle('Sample')

# out-of-sample
fig, axs = plt.subplots(1,3)
axs[0].imshow(oos)
axs[0].set_title("Original")
axs[1].imshow(k.transform_input(oos))
axs[1].set_title("Transformed")
im = axs[2].imshow(k.phi(x=oos))
axs[2].set_title("Feature Map")
fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
fig.suptitle('Out-of-Sample')

# kernel matrix
fig, axs = plt.subplots(2,2)
fig.suptitle('Kernel Matrix')

axs[0,0].imshow(k.K, vmin=-1, vmax=1)
axs[0,0].set_title("Sample - Sample")

axs[0,1].imshow(k.k(y=oos), vmin=-1, vmax=1)
axs[0,1].set_title("Sample - OOS")

axs[1,0].imshow(k.k(x=oos), vmin=-1, vmax=1)
axs[1,0].set_title("OOS - Sample")

im = axs[1,1].imshow(k.k(x=oos, y=oos), vmin=-1, vmax=1)
axs[1,1].set_title("OOS - OOS")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axs.ravel().tolist())