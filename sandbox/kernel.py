import kerch
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-1.2, 1.2, 50)
k = kerch.kernel.Epanechnikov(sample=x, sigma=1)

shape = k.k(x=0).squeeze()
plt.figure()
plt.plot(x, shape)
plt.title('Epanechnikov Shape')
plt.xlabel('x')
plt.ylabel('k(0,x)')
plt.show()