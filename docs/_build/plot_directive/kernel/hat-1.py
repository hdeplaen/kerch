import kerch
from matplotlib import pyplot as plt

k = kerch.kernel.hat(sample=range(10), lag=3)
plt.imshow(k.K)
plt.colorbar()
plt.title("Hat with lag " + str(k.lag))