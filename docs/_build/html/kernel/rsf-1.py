import kerch
import numpy as np
from matplotlib import pyplot as plt

x = np.random.randn(200, 2)

k = kerch.kernel.RSF(sample=x, num_weights=10)
phi = k.phi()
x_recon = k.explicit_preimage(phi)

plt.scatter(x[:,0], x[:,1], marker='*', label='original')
plt.scatter(x_recon[:,0], x_recon[:,1], marker='+', label='preimage')
plt.legend()