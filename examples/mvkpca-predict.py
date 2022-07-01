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

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "nystrom", "base_type": "rbf", "sample": x},
                       {"name": "time", "type": "nystrom", "base_type": "rbf", "sample": t},
                       dim_output=18)

mdl.solve(representation='primal')
# mdl.solve()

test = (max_t - min_t) * np.random.rand(100) + min_t
test = np.sort(test)
x_pred = mdl.predict_proj({"time": t}, method='closed')
x_real = mdl.view('space').phi(x)

from matplotlib import pyplot as plt

plt.scatter(x_pred[:, 2], x_pred[:, 1], c='r')
plt.scatter(x_real[:, 2], x_real[:, 1], c='b')
plt.show()

plt.figure()
plt.bar(range(len(mdl.vals)), mdl.vals)
plt.show()

print(x_pred)
print(x_real)

pass
