import kerch
import torch
from matplotlib import pyplot as plt

num, dim_input = 10,3

x = torch.randn(num, dim_input)
oos = torch.randn(num, dim_input)

k = kerch.kernel.Polynomial(sample=x, alpha=3, kernel_transform=['center', 'normalize'])

fig, axs = plt.subplots(2, 2)

axs[0,0].imshow(k.k(explicit=True))
axs[0,0].set_title("Explicit (sample)")

axs[0,1].imshow(k.k(explicit=False))
axs[0,1].set_title("Implicit (sample)")

axs[1,0].imshow(k.k(x=oos, y=oos, explicit=True))
axs[1,0].set_title("Implicit (out-of-sample)")

im = axs[1,1].imshow(k.k(x=oos, y=oos, explicit=False))
axs[1,1].set_title("Explicit (out-of-sample)")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')