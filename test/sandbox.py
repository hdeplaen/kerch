import kerpy as kp
import numpy as np
import logging

kp.set_log_level(logging.DEBUG)

k = kp.kernel.factory(type='rbf')
l = kp.model.LSSVM(kernel=k, representation="dual")

X = np.random.randn(10,4)
y = np.random.randn(10,2)
l.solve(X, y)