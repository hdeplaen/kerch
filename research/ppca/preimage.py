import kerch
from kerch.dataset import factory

x_train, _, x_test = factory("usps", 200, 0, 50)