import kerch
from matplotlib import pyplot as plt

k = kerch.kernel.indicator(sample=range(10), lag=3)
plt.imshow(k.K)
plt.colorbar()
plt.title("Indicator with lag " + str(k.lag))