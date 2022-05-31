import rkm
import torch

k = rkm.kernel.factory(type="rbf", sample=range(10))


print(k.K)