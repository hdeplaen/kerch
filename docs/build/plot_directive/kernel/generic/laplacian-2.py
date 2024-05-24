import kerch
import numpy as np
from matplotlib import pyplot as plt

x = np.sin(np.arange(50) / np.pi)

# automatic bandwidth with heuristic
k_laplacian = kerch.kernel.Laplacian(sample=x)
k_rbf = kerch.kernel.RBF(sample=x)

fig1, axs1 = plt.subplots(1, 2)

axs1[0].imshow(k_laplacian.K)
axs1[0].set_title(f"Laplacian ($\sigma$={k_laplacian.sigma:.2f})")

im1 = axs1[1].imshow(k_rbf.K)
axs1[1].set_title(f"RBF ($\sigma$={k_rbf.sigma:.2f})")

fig1.colorbar(im1, ax=axs1.ravel().tolist(), orientation='horizontal')

#  unity bandwidth
k_laplacian_sigma1 = kerch.kernel.Laplacian(sample=x, sigma=1)
f_rbf_sigma1 = kerch.kernel.RBF(sample=x, sigma=1)

fig2, axs2 = plt.subplots(1, 2)

axs2[0].imshow(k_laplacian_sigma1.K)
axs2[0].set_title(f"Laplacian ($\sigma$={k_laplacian_sigma1.sigma})")

im2 = axs2[1].imshow(f_rbf_sigma1.K)
axs2[1].set_title(f"RBF ($\sigma$={f_rbf_sigma1.sigma})")

fig2.colorbar(im2, ax=axs2.ravel().tolist(), orientation='horizontal')