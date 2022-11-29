import numpy as np
import kerch
from logging import DEBUG, INFO, WARNING
import torch
from matplotlib import pyplot as plt

# PRELIMINARIES
kerch.set_log_level(WARNING)

# DATA
NUM_POINTS = 1000

fun = lambda x: np.sin(
    x)  # np.concatenate([np.expand_dims(np.sin(x),1), np.expand_dims(x**2,1)], axis=1) # np.sin(x)  # (x-2*np.pi)**2

min_t = -5
max_t = 5
t = np.linspace(min_t, max_t, NUM_POINTS)
x = fun(t)

# MODEL
mdl = kerch.rkm.multiview.MVKPCA({"name": "space", "type": "rbf", "base_type": "rbf", "center": True, "sample": x},
                       {"name": "time", "type": "nystrom", "base_type": "rbf", "center": True, "sample": t},
                       dim_output=100)
mdl.solve(representation='primal')

# EXTRAPOLATE
test = (max_t - min_t) * np.random.rand(NUM_POINTS) + min_t
test = np.sort(test)
x_pred = mdl.predict_oos({"time": t}).detach()
# x_pred = mdl.predict_sample('time').detach()
x_real = mdl.view('space').phi(x).detach()
res = torch.norm(x_pred - x_real)

# PLOT
plt.figure()
plt.plot(t,x)
plt.show()

plt.figure()
plt.bar(range(len(mdl.vals)), mdl.vals)
plt.show()

plt.figure()
for num in range(NUM_POINTS-1,NUM_POINTS):
    plt.scatter(x_pred[0:num, 0], x_pred[0:num, 1], c='r')
    plt.scatter(x_real[0:num, 0], x_real[0:num, 1], c='b')
    plt.show()

# fig.show()
# plt.figure()
# plt.scatter(x_real[:, 2], x_real[:, 1], c='b')

# print(x_pred)
# print(x_real)

pass
