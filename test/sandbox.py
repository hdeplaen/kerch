import numpy as np
import kerch

kerch.set_log_level(0)

fun = lambda x: np.sin(x)  # np.sin(x)  # (x-2*np.pi)**2

min_t = -5
max_t = 5
t = np.linspace(min_t, max_t, 20)
x = fun(t)

#####################

mdl = kerch.rkm.MVKPCA({"name": "space", "name": "linear", "sample": x},
                       {"name": "time", "name": "rbf", "sample": t, "sigma": 1.},
                       representation='dual')

mdl.solve()

print('euclidean')
for p in mdl.manifold_parameters():
    print(p)

print('stiefel')
for p in mdl.manifold_parameters(type='stiefel'):
    print(p)

print('slow')
for p in mdl.manifold_parameters(type='slow'):
    print(p)

pass
