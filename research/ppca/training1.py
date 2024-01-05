import kerch
import kerch.data
from matplotlib import pyplot as plt

## PRELIMINARIES
N = 50
(x, y), _, _ = kerch.dataset.two_moons(N, 0)

k = kerch.kernel.RBF(center=True, sample=x)
print(k.sigma)