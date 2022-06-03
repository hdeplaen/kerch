import rkm
import torch
import logging

rkm.set_log_level(logging.INFO)

k = rkm.kernel.factory(type='rbf')
print(k.K)
k.dim_input = 2
k.num_sample = 10
print(k.K)