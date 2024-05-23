import kerch
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import math



torch.manual_seed(95)
newline = '\n'

# preliminaries
kernel_type = "rfstacked"
alpha = 1
beta = 2
gamma=1
delta=1
num_weights = [20, 100, 500]
num_components = 6
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


# plot h
def plot_individual(ax: plt.Axes, vals: torch.Tensor, title: str = ""):
    vals = vals.squeeze()
    assert len(vals.shape) == 1, 'Incorrect shape to be plotted.'
    mean = torch.mean(vals)
    std = torch.std(vals)

    # hist
    ax.hist(vals, bins=15, density=True, alpha=0.5, color='k')

    # pdf
    fact = 1 / (std * math.sqrt(2 * math.pi))
    xmin, xmax = ax.get_xlim()
    x = torch.linspace(xmin, xmax, 100)
    y = fact * torch.exp(-.5*(x-mean)**2 / (std**2) )
    ax.plot(x, y, 'k', linewidth=2)

    # labels
    ax.set_title(title + newline + f"($\mu$={mean:1.2f}, $\sigma$={std:1.2f})")
    # ax.set_xlabel("Value")
    # ax.set_ylabel("PDF")

def plot_vals(vals: torch.Tensor, title: str=""):
    vals = vals.squeeze()
    if len(vals.shape) == 1:
        vals = vals[:, None]
    assert len(vals.shape) == 2, 'Incorrect shape, must be of dimension 2.'

    num_plots = vals.shape[1]
    num_columns = math.ceil(num_plots ** (.5))
    num_rows = math.ceil(num_plots / num_columns)
    fig, axs = plt.subplots(num_rows, num_columns)

    for i, ax in enumerate(axs.ravel()):
        try:
            plot_individual(ax, vals[:,i], f"Component {i+1}")
        except IndexError:
            break
    # fig.suptitle(title)
    fig.tight_layout(pad=.5)
    return fig

plot_vals(h_sample, "Sample h").savefig('sample.png', format='png')
plot_vals(h_star, 'Generated h').savefig('gen.png', format='png')

# eigenvalues histogram
mpl.rcParams['hatch.linewidth'] = 1.5
fig, ax = plt.subplots()
vals = 100*kpca.vals / kpca.total_variance(normalize=False)
ax.bar(range(kpca.dim_output), vals, facecolor='none', edgecolor='k', hatch="///", linewidth=1.5)
ax.set_ylim(0,100)
ax.set_xlim(-.7,5.7)
ax.axvline(x=1.5, color='k', linestyle='dashed', linewidth=2)
ax.annotate(f"{vals[:2].sum():1.2f}%", xy=(0,85))
ax.annotate(f"{vals[2:].sum():1.2f}%", xy=(3,85))
ax.fill([1.5,1.5,5.7,5.7],[0,100,100,0], facecolor='k', edgecolor='none', alpha=.2)
# fig.suptitle("Eigenvalues")
ax.set_ylabel("Explained variance [%]")
ax.set_xticks(range(kpca.dim_output), [f"$\lambda_{i+1}$" for i in range(kpca.dim_output)])
fig.savefig("eigenvalues.png", format='png')


# plot_vals(sample_recon, 'Sample Phi (recon.)').savefig('sample.png', format='png')
# plot_vals(oos_recon, 'Generated Phi (recon.)').savefig('gen.png', format='png')



