import rkm
import torch
import logging

rkm.set_log_level(logging.DEBUG)

k = rkm.kernel.factory(type='rbf')
p = torch.nn.Parameter(torch.empty(10, 2))
v = rkm.model.primal_view(weights=p, kernel=k, sample=range(10))
v.init_sample(sample=range(10))
