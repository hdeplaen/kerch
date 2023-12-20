import kerch
import torch

n, d = 5, 4
x = torch.randn((n, d))
PPCA = kerch.level.PPCA(type='linear', sample=x, dim_output=3)
PPCA.solve()
k = PPCA.draw_k()