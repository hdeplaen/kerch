import kerch
import numpy as np
from matplotlib import pyplot as plt

x = np.sin(np.arange(50) / np.pi)

k_cos = kerch.kernel.Cosine(sample=x)
k_lin = kerch.kernel.Linear(sample=x, kernel_transform=['normalize'])

fig, axs = plt.subplots(1, 2)

axs[0].imshow(k_cos.K)
axs[0].set_title("Cosine")

im = axs[1].imshow(k_lin.K)
axs[1].set_title("Normalized Linear")

fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
