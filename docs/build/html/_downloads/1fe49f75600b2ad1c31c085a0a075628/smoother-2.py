import kerch
import torch
from matplotlib import pyplot as plt

# domain
x = torch.linspace(-3, 3, 500)

# define the kernels
k_l1 = kerch.kernel.Laplacian(sample=x, sigma=1)
k_l2 = kerch.kernel.Laplacian(sample=x, sigma=2)
k_t1 = kerch.kernel.Triweight(sample=x, sigma=1)
k_t2 = kerch.kernel.Triweight(sample=x, sigma=2)

# plot the shapes
plt.plot(x, k_l1.k(y=0).squeeze(), label=f"Laplacian with $\sigma$={k_l1.sigma}", color='black')
plt.plot(x, k_l2.k(y=0).squeeze(), label=f"Laplacian with $\sigma$={k_l2.sigma}", color='black', linestyle='dashed')
plt.plot(x, k_t1.k(y=0).squeeze(), label=f"Triweight with $\sigma$={k_t1.sigma}", color='red')
plt.plot(x, k_t2.k(y=0).squeeze(), label=f"Triweight with $\sigma$={k_t2.sigma}", color='red', linestyle='dashed')

# annotate the plot
plt.title('Kernel Shape')
plt.xlabel('x')
plt.ylabel('k(x,y=0)')
plt.ylim(-.25, 1.1)
plt.legend(loc='lower center', ncol=2)