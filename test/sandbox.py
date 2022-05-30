import rkm

k = rkm.kernel.factory(type="yolo", sample=range(10))
print(k.K)