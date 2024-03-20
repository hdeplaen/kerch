from __future__ import annotations

import kerch
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import math

kerch.set_ftype(torch.float64)
kerch.set_eps(1.e-15)

torch.manual_seed(95)
newline = '\n'

# preliminaries
kernel_type = "rfstacked"
alpha = 1
beta = 2
gamma=1
delta=1
num_weights = [10, 20, 50, 100, 150, 200]
num_components = 200
num_data = 100
num_draw = 100
sigma = .1
noise = .1
separation = [3, -3] # [1, .5]

# model
data = kerch.data.TwoMoons(num_training=num_data, noise=noise, separation=separation)
sample_original, labels = data.training_set[:]

# training
kpca = kerch.level.KPCA(kernel_type=kernel_type,
                        num_weights=num_weights,
                        sample=sample_original,
                        dim_output=num_components,
                        sample_transform=['standardize'],
                        kernel_transform=['center'],
                        representation='primal',
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        delta=delta,
                        sigma=sigma)
kpca.solve()

# reconstruction
sample_preimage = kpca.explicit_preimage(kpca.Phi)
h_sample = kpca.h(phi=kpca.Phi)
phi_sample = kpca.phi_map(h_sample)
sample_recon = kpca.explicit_preimage(phi_sample)

# # generation
h_star = kpca.draw_h(num_draw)
phi_star = kpca.phi_map(h_star)
oos_recon = kpca.explicit_preimage(phi_star)


# plot
fig, ax = plt.subplots()
ax.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')

ax.scatter(sample_recon[:, 0], sample_recon[:, 1], color='cyan', label=f"Reconstructed", marker='x')
ax.scatter(oos_recon[:, 0], oos_recon[:, 1], color='red', label=f"Generated", marker='.')

fig_range = data.info["range"]

ax.legend()
# fig.suptitle('Two Moons')
plt.xlim(fig_range[0], fig_range[1])
plt.ylim(fig_range[2], fig_range[3])
fig.savefig('input.png', format='png')

kerch.plot.matplotlib.plot_vals(h_sample, 6, "Sample h").savefig('sample.png', format='png')
kerch.plot.matplotlib.plot_vals(h_star, 6, 'Generated h').savefig('gen.png', format='png')
kerch.plot.matplotlib.plot_eigenvalues(kpca, labels=True, num_vals=6, section_div=2).savefig("eigenvalues.png",
                                                                                             format='png')

# kerch.plot.matplotlib.plot_vals(sample_recon, 'Sample Phi (recon.)').savefig('sample.png', format='png')
# kerch.plot.matplotlib.plot_vals(oos_recon, 'Generated Phi (recon.)').savefig('gen.png', format='png')
