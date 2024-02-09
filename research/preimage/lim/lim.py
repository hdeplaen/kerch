import kerch
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl

torch.manual_seed(95)  # for reproducibility

# preliminaries
num_sample, dim_sample = 50, 5
sample = torch.randn(num_sample, dim_sample)
kernel_type = "rfarcsinh"

alpha = 1

min, max = -3, 3


# plot
plt.rcParams['svg.fonttype'] = 'none'
def plot(k):
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(k.K, cmap='CMRmap', vmin=min, vmax=max)
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar()
    fig.tight_layout()
    plt.savefig(f"{kernel_type}-{k.num_weights}-{alpha}.svg", format='svg')
    plt.show()


# model
for n in [50, 500, 5000]:
    k = kerch.kernel.factory(kernel_type=kernel_type, sample=sample, num_weights=n, alpha=alpha)
    plot(k)

fig = plt.figure(figsize=(1, 5))
ax = fig.gca()
cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap='CMRmap',
                               norm=mpl.colors.Normalize(vmin=min, vmax=max))
fig.tight_layout()
plt.savefig(f"{kernel_type}-colorbar-{alpha}.svg", format='svg')
fig.show()
