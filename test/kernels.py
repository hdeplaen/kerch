from matplotlib import pyplot as plt
import rkm
import numpy as np

x = np.sin(np.arange(50) / np.pi) + 1.5
k = rkm.kernel.cosine(sample=x, center=True)
plt.imshow(k.K)
plt.show()