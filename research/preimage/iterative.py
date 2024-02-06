import kerch
from matplotlib import pyplot as plt

# preliminaries
num_components = 30
num_data = 30
num_draw = 5
num_iter = 10000

# model
data = kerch.data.TwoMoons(num_training=num_data, num_validation=50, num_test=50, noise=.02)
sample_original, labels = data.training_set[:]

kpca = kerch.level.KPCA(kernel_type='rbf', sample=sample_original, dim_output=num_components, kernel_transform=['center'])
kpca.solve()

h_star = kpca.draw_h(num_draw)
k_star = kpca.k_map(h_star)

oos_recon = kpca.implicit_preimage(k_star, method='iterative', verbose=True, num_iter=num_iter, lr=1.e-1)
sample_recon = kpca.implicit_preimage(kpca.K, method='iterative', verbose=True, num_iter=num_iter, lr=1.e-1)


# plot
fig, ax = plt.subplots()
ax.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')

ax.scatter(sample_recon[:, 0], sample_recon[:, 1], color='blue', label=f"Reconstructed", marker='x')
ax.scatter(oos_recon[:, 0], oos_recon[:, 1], color='red', label=f"Generated", marker='.')

ax.legend()
fig.suptitle('Reconstruction')
plt.show()