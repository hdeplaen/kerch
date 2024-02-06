import kerch
from matplotlib import pyplot as plt

# preliminaries
num_components = 5
num_data = 100
num_draw = 100

# model
data = kerch.data.TwoMoons(num_training=num_data, num_validation=50, num_test=50, noise=.1)
sample_original, labels = data.training_set[:]

kpca = kerch.level.KPCA(kernel_type='rbf', sample=sample_original, dim_output=2, kernel_transform=['center'])
kpca.solve()

h_star = kpca.draw_h(num_draw)
k_star = kpca.k_map(h_star)

# KNN
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')
ax2.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')

for k, c in zip([1, 2, 5], ['red', 'green', 'blue']):
    sample_recon = kpca.implicit_preimage(kpca.K, method='knn', num=k)
    oos_recon = kpca.implicit_preimage(k_star, method='knn', num=k)
    ax1.scatter(sample_recon[:, 0], sample_recon[:, 1], color=c, label=f"k={k}", marker='.')
    ax2.scatter(oos_recon[:, 0], oos_recon[:, 1], color=c, label=f"k={k}", marker='.')

ax1.legend()
ax2.legend()
fig1.suptitle('Reconstruction')
fig2.suptitle('Generation')
plt.show()


# SMOOTHER
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')
ax2.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')

for k, c in zip([5, 20, 50], ['red', 'green', 'blue']):
    sample_recon = kpca.implicit_preimage(kpca.K, method='smoother', num=k)
    oos_recon = kpca.implicit_preimage(k_star, method='smoother', num=k)
    ax1.scatter(sample_recon[:, 0], sample_recon[:, 1], color=c, label=f"k={k}", marker='.')
    ax2.scatter(oos_recon[:, 0], oos_recon[:, 1], color=c, label=f"k={k}", marker='.')

ax1.legend()
ax2.legend()
fig1.suptitle('Reconstruction')
fig2.suptitle('Generation')
plt.show()


# ITERATIVE
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')
ax2.scatter(sample_original[:, 0], sample_original[:, 1], color='black', label='Original', marker='+')

for k, c in zip([5], ['red']):
    sample_recon = kpca.implicit_preimage(kpca.K, method='iterative')
    oos_recon = kpca.implicit_preimage(k_star, method='iterative')
    ax1.scatter(sample_recon[:, 0], sample_recon[:, 1], color=c, label=f"Iterative", marker='.')
    ax2.scatter(oos_recon[:, 0], oos_recon[:, 1], color=c, label=f"Iterative", marker='.')

ax1.legend()
ax2.legend()
fig1.suptitle('Reconstruction')
fig2.suptitle('Generation')
plt.show()

