import kerch
import torch
from matplotlib import pyplot as plt

torch.manual_seed(95)

# preliminaries
kernel_type = "rflrelu"
alpha = .01
num_weights = 5000
num_components = 4
num_data = 100
num_draw = 100

# model
data = kerch.data.TwoMoons(num_training=num_data, noise=.1)
sample_original, labels = data.training_set[:]

# training
kpca = kerch.level.KPCA(kernel_type=kernel_type,
                        num_weights=num_weights,
                        sample=sample_original,
                        dim_output=num_components,
                        sample_transform=['standardize'],
                        kernel_transform=['center'],
                        representation='primal',
                        alpha=.1)
kpca.solve()

# reconstruction
sample_recon = kpca.explicit_preimage(kpca.Phi)
h_sample = kpca.h_map(phi=kpca.Phi)

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
fig.suptitle('Two Moons')
# plt.xlim(fig_range[0], fig_range[1])
# plt.ylim(fig_range[2], fig_range[3])
plt.show()


# plot h
