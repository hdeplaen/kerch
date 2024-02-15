import matplotlib.pyplot as plt

import kerch
import torch
import torchvision
import numpy as np
from math import sqrt

kerch.set_ftype(torch.double)



# preliminaries
kernel_type = 'rflrelu'
alpha = .1
num_data = 60000
num_weights = 4000
num_components = 30
grid_size = 8
num_draw = grid_size ** 2
fig_size = 28
sigma=.2

# dataset
mnist = torchvision.datasets.MNIST('./files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()]))
sampler = torch.utils.data.RandomSampler(mnist, replacement=True)
loader = torch.utils.data.DataLoader(mnist, batch_size=num_data, sampler=sampler)
sample, _ = next(iter(loader))
sample = sample.view(sample.shape[0], -1)

# model
kpca = kerch.level.KPCA(kernel_type=kernel_type,
                        num_weights=num_weights,
                        sample=sample,
                        dim_output=num_components,
                        sample_transform=['standardize'],
                        kernel_transform=['center'],
                        representation='primal',
                        alpha=alpha,
                        sigma=sigma)
kpca.solve()
print('Solved')

def pre_image(input):
    vals = kpca.explicit_preimage(input)
    # vals /= vals.max(dim=1, keepdim=True).values
    return vals

sample_preimage = pre_image(kpca.Phi)[:num_draw,:].reshape(-1, fig_size, fig_size)
h_sample = kpca.h(phi=kpca.Phi)
phi_recon = kpca.phi_map(h_sample)
sample_recon = pre_image(phi_recon)[:num_draw,:].reshape(-1, fig_size, fig_size)
h_star = kpca.draw_h(num_draw)
phi_star = kpca.phi_map(h_star)
oos_recon = pre_image(phi_star).reshape(-1, fig_size, fig_size)
sample_original = sample[:num_draw, :].reshape(-1, fig_size, fig_size)


# original
def plot(val):
    fig, ax = plt.subplots()
    grid_img = torchvision.utils.make_grid(val.unsqueeze(1), nrow=grid_size)
    ax.imshow(grid_img.permute((1,2,0)).clip(0, 1))
    ax.set_axis_off()
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    return fig

plot(sample_original).savefig(f"original.png", format='png')
plot(sample_preimage).savefig(f"preimage.png", format='png')
plot(sample_recon).savefig(f"reconstruction.png", format='png')
plot(oos_recon).savefig(f"generation.png", format='png')

# interpolation


lambda_range = np.linspace(0, 1, 10)

fig, axs = plt.subplots(1, 10, figsize=(30, 3))
fig.subplots_adjust(wspace=.1)
axs = axs.ravel()
for ind, l in enumerate(lambda_range):
    l = float(l)
    inter_latent = l * h_sample[1,:] + (1 - l) * h_sample[2,:]
    inter_image = kpca.phi_map(inter_latent[None, :])
    inter_image = pre_image(inter_image)
    inter_image = inter_image.reshape(fig_size, fig_size)
    image = inter_image.clip(0,1).numpy()

    axs[ind].imshow(image, cmap='gray')
    axs[ind].set_title('$\lambda$=' + str(round(l, 1)))
    axs[ind].axis('off')
# fig.show()
fig.savefig('interp_original.png', format='png')


fig, axs = plt.subplots(1, 10, figsize=(30, 3))
fig.subplots_adjust(wspace=.1)
axs = axs.ravel()
for ind, l in enumerate(lambda_range):
    l = float(l)
    inter_latent = l * kpca.Phi[1,:] + (1 - l) * kpca.Phi[2,:]
    inter_image = pre_image(inter_latent[None, :])
    inter_image = inter_image.reshape(fig_size, fig_size)
    image = inter_image.clip(0,1).numpy()

    axs[ind].imshow(image, cmap='gray')
    axs[ind].set_title('$\lambda$=' + str(round(l, 1)))
    axs[ind].axis('off')
# fig.show()
fig.savefig('interp_recon.png', format='png')



