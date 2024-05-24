import kerch
import torch
from matplotlib import pyplot as plt

# data
fun = lambda x: torch.sin(x ** 2)

x_equal = torch.linspace(0, 2, 100)
x_nonequal = 2 * torch.sort(torch.rand(40)).values

y_original = fun(x_equal)
y_noisy = fun(x_nonequal) + .2 * torch.randn_like(x_nonequal)

plt.plot(x_equal, y_original, label="Original Data", color="black", linestyle='dotted')
plt.scatter(x_nonequal, y_noisy, label="Noisy Data", color="black")

# kernels
kernels = [('RBF', 'red'),
           ('Laplacian', 'orange'),
           ('Logistic', 'olive'),
           ('Epanechnikov', 'gold'),
           ('Quartic', 'chartreuse'),
           ('Silverman', 'green'),
           ('Triangular', 'teal'),
           ('Tricube', 'cyan'),
           ('Triweight', 'royalblue'),
           ('Uniform', 'purple')]

# kernel smoother
for name, c in kernels:
    y_reconstructed = kerch.method.kernel_smoother(domain=x_nonequal, observations=y_noisy, kernel_type=name.lower())
    plt.plot(x_nonequal, y_reconstructed, label=name, color=c)

# plot
plt.title('Kernel Smoothing')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower center', ncol=3)