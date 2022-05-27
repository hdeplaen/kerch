import rkm
from matplotlib import pyplot as plt

x = range(10)

k1 = rkm.kernel.indicator(sample=x, lag=3)
k2 = rkm.kernel.hat(sample=x, lag=3)
k3 = rkm.kernel.rbf(sample=x, sigma=3)

plt.figure(1)
plt.imshow(k1.K)
plt.colorbar()
plt.title("Indicator with lag " + str(k1.lag))

plt.figure(2)
plt.imshow(k2.K)
plt.colorbar()
plt.title("Hat with lag " + str(k2._lag))

plt.figure(3)
plt.imshow(k3.K)
plt.colorbar()
plt.title("RBF with sigma " + str(k3.sigma))

plt.show()