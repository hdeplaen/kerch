import kerch
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-5, 5, 200)
k_squared = kerch.kernel.Exponential(sample=x, sigma=1)                     # same as RBF kernel
k_non_squared = kerch.kernel.Exponential(sample=x, sigma=1, squared=False)  # same as Laplacian kernel
shape = torch.cat((k_squared.k(y=0), k_non_squared.k(y=0)), dim=1)

plt.figure()
plt.plot(x, shape)
plt.title('Exponential Shape')
plt.legend(['Squared (default)', 'Non-Squared'])
plt.xlabel('x')
plt.ylabel('k(x,y=0)')