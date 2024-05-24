import kerch
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-1.2, 1.2, 100)
k = kerch.kernel.Tricube(sample=x, sigma=1)
shape = k.k(y=0).squeeze()

plt.figure()
plt.plot(x, shape)
plt.title('Tricube Shape')
plt.xlabel('x')
plt.ylabel('k(x,y=0)')