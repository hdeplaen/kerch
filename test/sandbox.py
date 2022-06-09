import kerpy as kp
import numpy as np
import logging

kp.set_log_level(logging.INFO)

l = kp.model.LSSVM(kernel_type="rbf", representation="dual")

X = np.random.randn(10,4)
y = np.random.randn(10,1)

l.set_data_prop(X, y, proportions=[.8, .2, 0])
l.hyperopt({"sigma", "gamma"})