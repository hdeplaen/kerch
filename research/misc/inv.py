import torch
import kerch

N = 50
dim = 2
feat = 500

x1 = torch.randn((N, dim)) + 2
x2 = torch.randn((N, dim)) - 2
x = torch.cat((x1,x2), dim=0)

k = kerch.kernel.generic.random_features(sample=x, num_weights=feat)
phi = k.phi()
x_tile = k.phi_pinv()

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.scatter(x_tile[:,0],x[:,1])
plt.show()