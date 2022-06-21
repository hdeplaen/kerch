import numpy as np
import kerch

fun = lambda x: np.sin(x)  # (x-2*np.pi)**2

t = np.linspace(0, 4 * np.pi, 50)
x = fun(t)

from matplotlib import pyplot as plt

plt.plot(t, x)
plt.show()

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "rbf", "sample": x},
                       {"name": "time", "type": "rbf", "sample": t, "sigma": 1.},
                       dim_output=35)

mdl.solve()

test = np.linspace(0, 4 * np.pi, 100)
# print(np.sin(test))
sol = mdl.predict({"time": test}, tot_iter=500, lr=1.e+2)
print(sol)

plt.plot(test, sol['space'].data)
plt.plot(test, fun(test))
plt.show()
