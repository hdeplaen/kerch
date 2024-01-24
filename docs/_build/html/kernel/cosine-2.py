import kerch
import numpy as np
from matplotlib import pyplot as plt

sin = np.expand_dims(np.sin(np.arange(50) / np.pi), axis=1)
log = np.expand_dims(np.sin(np.log(np.arange(50)+1)), axis=1)

x1 = sin
x2 = np.concatenate((sin,log), axis=1)

k1 = kerch.kernel.Cosine(sample=x1)
k2 = kerch.kernel.Cosine(sample=x2)

fig, axs = plt.subplots(1, 2)

axs[0].imshow(k1.K)
axs[0].set_title("One Dimension")

im = axs[1].imshow(k2.K)
axs[1].set_title("Two Dimensions")

fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')