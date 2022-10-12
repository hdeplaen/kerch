import numpy as np
import kerch
from logging import DEBUG, INFO

kerch.set_log_level(INFO)

fun = lambda x: np.sin(
    x)  # np.concatenate([np.expand_dims(np.sin(x),1), np.expand_dims(x**2,1)], axis=1) # np.sin(x)  # (x-2*np.pi)**2

min_t = -5
max_t = 5
t = np.linspace(min_t, max_t, 20)
x = fun(t)

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "nystrom", "base_type": "rbf", "center": False, "sample": x},
                       {"name": "time", "type": "nystrom", "base_type": "rbf", "center": False, "sample": t},
                       dim_output=25)

mdl.solve(representation='primal')
# mdl.solve()

test = (max_t - min_t) * np.random.rand(100) + min_t
test = np.sort(test)
x_pred = mdl.predict_oos({"time": t}).detach()
x_real = mdl.view('space').phi(x).detach()

import torch
res = torch.norm(x_pred - x_real)

from matplotlib import pyplot as plt

plt.figure()
plt.bar(range(len(mdl.vals)), mdl.vals)
plt.show()

plt.figure()
for num in range(1,20):
    plt.scatter(x_pred[0:num, 0], x_pred[0:num, 1], c='r')
    plt.scatter(x_real[0:num, 0], x_real[0:num, 1], c='b')
    plt.show()

# plt.figure()
# plt.scatter(x_real[:, 2], x_real[:, 1], c='b')

# print(x_pred)
# print(x_real)

pass
