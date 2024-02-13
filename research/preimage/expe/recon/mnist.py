import matplotlib.pyplot as plt

import kerch
import torch
import torchvision
import numpy as np
from math import sqrt

kerch.set_ftype(torch.double)



# preliminaries
kernel_type = 'rfarcsinh'
alpha = .05
num_data = 20000
num_weights = 4000
num_components = 10
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

def pre_image(input):
    vals = kpca.explicit_preimage(input)
    # vals /= vals.max(dim=1, keepdim=True).values
    return vals

sample_recon = pre_image(kpca.Phi)[:num_draw,:].reshape(-1, fig_size, fig_size)
h_star = kpca.draw_h(num_draw)
phi_star = kpca.phi_map(h_star)
oos_recon = pre_image(phi_star).reshape(-1, fig_size, fig_size)
sample_original = sample[:num_draw, :].reshape(-1, fig_size, fig_size)


# original
def plot(val):
    grid_img = torchvision.utils.make_grid(val.unsqueeze(1), nrow=grid_size)
    plt.imshow(grid_img.permute((1,2,0)).clip(0, 1))
    plt.show()

plot(sample_original)
plot(sample_recon)
plot(oos_recon)

# interpolation

def interpolation(lambda1, latent_1, latent_2):
    # interpolation of the two latent vectors
    inter_latent = lambda1 * latent_1 + (1 - lambda1) * latent_2

    # reconstruct interpolated image
    inter_image = pre_image(inter_latent[None, :])
    inter_image = inter_image

    return inter_image.reshape(fig_size, fig_size)

lambda_range = np.linspace(0, 1, 10)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(wspace=.1)
axs = axs.ravel()

for ind, l in enumerate(lambda_range):
    inter_image = interpolation(float(l), kpca.Phi[1,:], kpca.Phi[2,:])

    image = inter_image.clip(0,1).numpy()

    axs[ind].imshow(image, cmap='gray')
    axs[ind].set_title('$\lambda$=' + str(round(l, 1)))
    axs[ind].axis('off')
plt.show()






