import kerch
import numpy as np
from matplotlib import pyplot as plt

x = np.sin(np.arange(50) / np.pi)

k1 = kerch.kernel.Polynomial(sample=x, alpha=2, beta=1)
k2 = kerch.kernel.Polynomial(sample=x, alpha=2, beta=5)
k3 = kerch.kernel.Polynomial(sample=x, alpha=5, beta=1)
k4 = kerch.kernel.Polynomial(sample=x, alpha=5, beta=5)

fig, axs = plt.subplots(2, 2)

axs[0,0].imshow(k1.K)
axs[0,0].set_title(f"Alpha = {k1.alpha}, Beta = {k1.beta}")

axs[0,1].imshow(k2.K)
axs[0,1].set_title(f"Alpha = {k2.alpha}, Beta = {k2.beta}")

axs[1,0].imshow(k3.K)
axs[1,0].set_title(f"Alpha = {k3.alpha}, Beta = {k3.beta}")

im = axs[1,1].imshow(k4.K)
axs[1,1].set_title(f"Alpha = {k4.alpha}, Beta = {k4.beta}")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')