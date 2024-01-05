import kerch
import torch

n, d = 50, 10
x = torch.randn((n, d))
t = torch.randn((n, 1))

x_test = torch.randn((n,d))
t_test = torch.randn((n,1))

rkm = kerch.rkm.RKM()
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=5)
rkm.append_level(level_type='kpca', constraint='soft', representation='dual', dim_output=2)
rkm.append_level(level_type='lssvm', constraint='soft', representation='dual', dim_output=1)

trainer = kerch.rkm.Trainer(model=rkm)
trainer.train_data = x
trainer.train_labels = t
trainer.test_data = x_test
trainer.test_labels = t_test