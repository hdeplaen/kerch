import kerch
import numpy as np
from matplotlib import pyplot as plt

x = np.sin(np.arange(50) / np.pi)
plt.figure(0)
plt.plot(x)

k = kerch.kernel.polynomial(sample=x, degree=2)
plt.figure(1)
plt.imshow(k.K)
plt.colorbar()