import rkm
import torch
import logging

rkm.set_log_level(logging.INFO)

k = rkm.kernel.factory(type='rbf')
v = rkm.model.view(kernel=k)
v.init_sample(range(10))
print(v(5, representation="primal"))
