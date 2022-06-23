import numpy as np
import kerch

fun = lambda x: np.sin(x)  # np.sin(x)  # (x-2*np.pi)**2

min_t = -5
max_t = 5
t = np.linspace(min_t, max_t, 20)
x = fun(t)

from matplotlib import pyplot as plt

plt.figure(0)
plt.plot(t, x)
plt.show()

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "rbf", "sample": x},
                       {"name": "time", "type": "rbf", "sample": t, "sigma": 1.},
                       dim_output=10)

mdl.solve()

test = (max_t - min_t) * np.random.rand(100) + min_t
test = np.sort(test)
# print(np.sin(test))
sol = mdl.predict({"time": test}, tot_iter=1500, lr=.5)
print(sol)

plt.figure(1)
plt.plot(test, sol['space'].data)
plt.plot(test, fun(test))
plt.show()
