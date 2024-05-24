import kerch
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-5, 5, 200)
k_rbf = kerch.kernel.RBF(sample=x, sigma=1)
k_laplacian = kerch.kernel.Laplacian(sample=x, sigma=1)
shape = torch.cat((k_rbf.k(y=0), k_laplacian.k(y=0)), dim=1)

plt.figure()
plt.plot(x, shape)
plt.title('Kernel Shape')
plt.legend(['RBF',
            'Laplacian'])
plt.xlabel('x')
plt.ylabel('k(x,y=0)')