from .. import utils

@utils.kwargs_decorator({"kernel_type": "linear"})
def factory(**kwargs):
    kernel_type = kwargs["kernel_type"]
    switcher = {"linear": rkm.kernel.LinearKernel.linear,
                "rbf": rkm.kernel.RBFKernel.rbf,
                "explicit": rkm.kernel.ExplicitNNKernel.explicit_nn,
                "implicit": rkm.kernel.ImplicitNNKernel.implicit_nn,
                "polynomial": rkm.kernel.PolynomialKernel.polynomial,
                "sigmoid": rkm.kernel.SigmoidKernel.sigmoid,
                "indicator":  rkm.kernel.IndicatorKernel.indicator,
                "hat": rkm.kernel.HatKernel.hat,
                "nystrom": rkm.kernel.NystromKernel.nystrom}
    if kernel_type not in switcher:
        raise NameError("Invalid kernel type.")
    return switcher[kernel_type](**kwargs)

