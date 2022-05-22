import rkm
import numpy as np

x = np.random.randn(15,3)
k = rkm.kernel.rbf(sigma=1., sample=x)
K = k.K