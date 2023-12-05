import kerch
import numpy as np
from matplotlib import pyplot as plt

sample = np.sin(np.arange(0, 15) / np.pi) + .1  # sample
oos = np.sin(np.arange(15, 30) / np.pi) + .1  # out-of-sample

k = kerch.kernel.factory(type="rbf", sample=sample, kernel_projections=['center'])

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(k.K, vmin=-1, vmax=1)
axs[0, 0].set_title("Sample - Sample")

axs[0, 1].imshow(k.k(y=oos), vmin=-1, vmax=1)
axs[0, 1].set_title("Sample - OOS")

axs[1, 0].imshow(k.k(x=oos), vmin=-1, vmax=1)
axs[1, 0].set_title("OOS - Sample")

im = axs[1, 1].imshow(k.k(x=oos, y=oos), vmin=-1, vmax=1)
axs[1, 1].set_title("OOS - OOS")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axs.ravel().tolist())
plt.show()
