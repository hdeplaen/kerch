import rkm
import logging

rkm.logger.setLevel(logging.DEBUG)

k = rkm.kernel.factory(type="yolo")
print(k.K)
k.dim_input = 1
k.num_sample = 10
print(k.K)
