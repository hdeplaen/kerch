import kerch
import numpy as np
from matplotlib import pyplot as plt

x = np.sin(np.arange(50) / np.pi)
plt.figure(0)
plt.plot(x)

k = kerch.kernel.rbf(sample=x)

plt.figure(1)
plt.imshow(k.K)
plt.colorbar()
plt.title("Sigma = "+str(k.sigma))

k.sigma = 1

plt.figure(2)
plt.imshow(k.K)
plt.colorbar()
plt.title("Sigma = "+str(k.sigma))