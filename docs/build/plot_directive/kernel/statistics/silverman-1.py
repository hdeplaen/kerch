import kerch
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-6, 6, 100)
k = kerch.kernel.Silverman(sample=x, sigma=1)
shape = k.k(y=0).squeeze()

plt.figure()
plt.plot(x, shape)
plt.title('Silverman Shape')
plt.xlabel('x')
plt.ylabel('k(x,y=0)')