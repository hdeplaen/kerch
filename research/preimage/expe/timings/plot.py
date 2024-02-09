from matplotlib import pyplot as plt
import numpy as np

with np.load('mnist.npz') as data:
    timings_train, timings_recon, rel_var, num_components, num_weights, alpha, kernel_type = data.values()

# shape [type, weights, components, iter]
legends = ['Fourier', 'ReLU', 'LReLU', 'Identity', 'Arcinsh']
linestyles = [(5, (10, 3)), 'solid', 'dashdot', 'dotted', 'dashed']


# timing[w] fixed c (500)
tw_mean = np.mean(timings_train[:,:-1,-2,:], axis=2)
tw_std = np.std(timings_train[:,:-1,-2,:], axis=2)

fig = plt.figure()
for i in range(len(kernel_type)):
    plt.plot(num_weights[:-1], tw_mean[i,:], label=legends[i], linestyle=linestyles[i])
    plt.fill_between(num_weights[:-1], tw_mean[i,:] - tw_std[i,:], tw_mean[i,:] + tw_std[i,:], alpha=.2)
fig.gca().set_xscale('log')
fig.gca().set_yscale('log')
plt.xlabel('Number of components $q$')
plt.ylabel('Time [s]')
plt.legend()
plt.savefig('tw.svg', format='svg')
plt.show()


# timing[c] fixed w (10)
tc_mean = np.mean(timings_train[:,0,3:,:], axis=2)
tc_std = np.std(timings_train[:,0,3:,:], axis=2)

fig = plt.figure()
for i in range(len(kernel_type)):
    plt.plot(num_components[3:], tc_mean[i,:], label=legends[i], linestyle=linestyles[i])
    plt.fill_between(num_components[3:], tc_mean[i,:] - tc_std[i,:], tc_mean[i,:] + tc_std[i,:], alpha=.2)
fig.gca().set_xscale('log')
fig.gca().set_yscale('log')
plt.xlabel('Number of weights $s$')
plt.ylabel('Time [s]')
plt.legend()
plt.savefig('tc.svg', format='svg')
plt.show()