import kerch
from sklearn import datasets
from matplotlib import pyplot as plt

## EXAMPLE 1
x = datasets.make_moons(50)[0]                      # generate dataset

k1 = kerch.kernel.RBF(sample=x)                     # initiate kernel
x_recon_2 = k1.implicit_preimage(k1.K, knn=2)       # compute the different
x_recon_5 = k1.implicit_preimage(k1.K, knn=5)       # preimages with varying
x_recon_10 = k1.implicit_preimage(k1.K, knn=10)     # number of closest points
x_recon_25 = k1.implicit_preimage(k1.K, knn=25)     # used...
x_recon_50 = k1.implicit_preimage(k1.K, knn=50)

# plot
plt.figure(0)
plt.plot(x[:,0], x[:,1], 'k*', label='original and closest')
plt.plot(x_recon_2[:, 0], x_recon_2[:, 1], 'r*', label='2 closest')
plt.plot(x_recon_5[:,0], x_recon_5[:,1], 'm*', label='5 closest')
plt.plot(x_recon_10[:,0], x_recon_10[:,1], 'b*', label='10 closest')
plt.plot(x_recon_25[:,0], x_recon_25[:,1], 'c*', label='25 closest')
plt.plot(x_recon_50[:,0], x_recon_50[:,1], 'g*', label='50 closest (all)')
plt.title(f'RBF preimage with kernel smoother\n'
          f'Number of chosen points (sigma={k1.sigma:3.2f})')
plt.legend()
plt.show()

## EXAMPLE 2
x_noise = datasets.make_moons(250, noise=.1)[0] # generate dataset
k2 = kerch.kernel.RBF(sample=x_noise)           # initiate kernel
x_denoised = k2.implicit_preimage(k2.K, knn=25) # compute the preimage

# plot
plt.figure(1)
plt.plot(x_noise[:,0], x_noise[:,1], 'k.', label='original')
plt.plot(x_denoised[:,0], x_denoised[:,1], 'r.', label='denoised (25 closest)')
plt.title(f'RBF preimage with kernel smoother\n'
          f'Denoising (sigma={k2.sigma:3.2f})')
plt.legend()
plt.show()