import kerch
import torch

n, d = 10, 2
x = torch.randn((n,d))

rkm = kerch.rkm.RKM()
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=5)
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=4)
rkm.append_level(level_type='lssvm', constraint='soft', representation='dual', dim_output=3)
print(rkm)
rkm.sample = x
rkm.train()
rkm()