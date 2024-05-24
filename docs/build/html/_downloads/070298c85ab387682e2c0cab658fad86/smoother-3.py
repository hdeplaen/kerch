import kerch
import torch
from matplotlib import pyplot as plt

# data
fun = lambda x: torch.sin(x ** 2)

x_equal = torch.linspace(0, 2, 100)
x_nonequal = 2 * torch.sort(torch.rand(40)).values

y_original = fun(x_equal)
y_noisy = fun(x_nonequal) + .2 * torch.randn_like(x_nonequal)

# plot
fig, axs = plt.subplots(1, 2)
for ax in axs.flatten():
    ax.plot(x_equal, y_original, label="Original Data", color="black", linestyle='dotted')
    ax.scatter(x_nonequal, y_noisy, label="Noisy Data", color="black")
    plt.title('Kernel Smoothing')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# kernel smoother
sigmas = [(0.05, 'red'),
          (0.2, 'green'),
          (0.5, 'cyan'),
          (1.0, 'purple')]
for s, c in sigmas:
    y_laplacian = kerch.method.kernel_smoother(domain=x_nonequal, observations=y_noisy, kernel_type='laplacian', sigma=s)
    y_triweight = kerch.method.kernel_smoother(domain=x_nonequal, observations=y_noisy, kernel_type='triweight', sigma=s)
    axs[0].plot(x_nonequal, y_laplacian, color=c, label=f"Bandwidth $\sigma$={s}")
    axs[1].plot(x_nonequal, y_triweight, color=c)

# plot
fig.suptitle('Kernel Smoothing')
axs[0].set_title('Laplacian')
axs[1].set_title('Triweight')
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3)