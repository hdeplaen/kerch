import numpy as np
import kerch
from logging import DEBUG, INFO

kerch.set_log_level(INFO)

fun = lambda x: np.sin(x)  # np.sin(x)  # (x-2*np.pi)**2

min_t = -5
max_t = 5
t = np.linspace(min_t, max_t, 20)
x = fun(t)

from matplotlib import pyplot as plt

plt.figure(0)
plt.plot(t, x)
plt.show()

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "nystrom", "base_type": "rbf", "sample": x},
                       {"name": "time", "type": "nystrom", "base_type": "rbf", "sample": t},
                       dim_output=10)

mdl.solve(representation='primal')

test = (max_t - min_t) * np.random.rand(100) + min_t
test = np.sort(test)
# print(np.sin(test))
sol = mdl.predict({"time": test}, representation='primal')
print(sol)

plt.figure(1)
plt.plot(test, sol['space'].data)
plt.plot(test, fun(test))
plt.show()
