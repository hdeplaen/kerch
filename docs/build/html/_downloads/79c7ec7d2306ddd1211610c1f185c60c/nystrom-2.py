import kerch
import numpy as np
from matplotlib import pyplot as plt

sample = np.sin(np.arange(0,15) / np.pi) + .1
oos = np.sin(np.arange(15,30) / np.pi) + .1

k_base = kerch.kernel.RBF(sample=sample)
k = kerch.kernel.Nystrom(base_kernel=k_base, dim=6)

# kernel matrix
fig1, axs1 = plt.subplots(2,2)
fig1.suptitle('Kernel Matrices of the Base Kernel (RBF)')

axs1[0,0].imshow(k_base.K, vmin=0, vmax=1)
axs1[0,0].set_title("Sample - Sample")

axs1[0,1].imshow(k_base.k(y=oos), vmin=0, vmax=1)
axs1[0,1].set_title("Sample - OOS")

axs1[1,0].imshow(k_base.k(x=oos), vmin=0, vmax=1)
axs1[1,0].set_title("OOS - Sample")

im1 = axs1[1,1].imshow(k_base.k(x=oos, y=oos), vmin=0, vmax=1)
axs1[1,1].set_title("OOS - OOS")

for ax in axs1.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig1.colorbar(im1, ax=axs1.ravel().tolist())

# explicit feature map
fig2, axs2 = plt.subplots(1,2)
fig2.suptitle('Explicit Feature Maps (Nystrom)')

axs2[0].imshow(k.Phi)
axs2[0].set_title("Sample")

im2 = axs2[1].imshow(k.phi(x=oos))
axs2[1].set_title("OOS")

for ax in axs2.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig2.colorbar(im2, ax=axs2.ravel().tolist())

# kernel matrix from the explicit feature map
fig3, axs3 = plt.subplots(2,2)
fig3.suptitle('Kernel Matrices from the Explicit Feature Map (Nystrom)')

axs3[0,0].imshow(k.K, vmin=0, vmax=1)
axs3[0,0].set_title("Sample - Sample")

axs3[0,1].imshow(k.k(y=oos), vmin=0, vmax=1)
axs3[0,1].set_title("Sample - OOS")

axs3[1,0].imshow(k.k(x=oos), vmin=0, vmax=1)
axs3[1,0].set_title("OOS - Sample")

im3 = axs3[1,1].imshow(k.k(x=oos, y=oos), vmin=0, vmax=1)
axs3[1,1].set_title("OOS - OOS")

for ax in axs3.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig3.colorbar(im3, ax=axs3.ravel().tolist())