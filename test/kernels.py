import rkm
import numpy as np

x = np.sin(np.arange(50) / np.pi)
x = x[:,None]
k = rkm.kernel.cosine(sigma=1., sample=x)
K = k.K

from matplotlib import pyplot as plt
plt.imshow(K)
plt.show()