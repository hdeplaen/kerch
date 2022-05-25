import rkm
import numpy as np

x = np.sin(np.arange(50) / np.pi)
x = x[:,None]
k = rkm.kernel.indicator(sigma=1., sample=x, center=False, normalize=False)
K = k.K

from matplotlib import pyplot as plt
plt.imshow(K)
plt.show()