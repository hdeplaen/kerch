import kerch
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-5, 5, 100)
k = kerch.kernel.Logistic(sample=x, sigma=1)
shape = k.k(y=0).squeeze()

plt.figure()
plt.plot(x, shape)
plt.title('Logistic Shape')
plt.xlabel('x')
plt.ylabel('k(x,y=0)')