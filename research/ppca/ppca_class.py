import kerch
import torch

n, d = 5, 4
x = torch.randn((n, d))
PPCA = kerch.level.PPCA(type='linear', sample=x, dim_output=3)
PPCA.solve()
k_tilde = PPCA.draw_k(10)
x_tilde = PPCA.kernel.implicit_preimage(k_tilde, method='iterative')
print(x_tilde)