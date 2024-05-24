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

# we define the sample
t = np.expand_dims(np.arange(0,15), axis=1)
sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

# we also define an out_of_sample
t_oos = np.expand_dims(np.arange(15,30), axis=1)
oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

# we initialize our new kernel
k = MyImplicit(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['center','normalize'])

# sample
fig, axs = plt.subplots(1,2)
axs[0].imshow(sample)
axs[0].set_title("Original")
im = axs[1].imshow(k.current_sample_projected)
axs[1].set_title("Transformed")
fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
fig.suptitle('Sample')

# out-of-sample
fig, axs = plt.subplots(1,2)
axs[0].imshow(oos)
axs[0].set_title("Original")
im = axs[1].imshow(k.transform_input(oos))
axs[1].set_title("Transformed")
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