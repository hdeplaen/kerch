import kerch
import torch

n, d = 50, 10
x = torch.randn((n, d))
t = torch.randn((n, 1))

rkm = kerch.model.RKM()
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=5)
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=2)
rkm.append_level(level_type='lssvm', constraint='soft', representation='dual', dim_output=1)
print(rkm)

rkm.init_sample(x)
rkm.init_levels()
rkm.init_targets(t)

rkm.train()
print(rkm())
print(rkm)
