import rkm.src

@rkm.src.kwargs_decorator({"kernel_type": "linear"})
def factory(**kwargs):
    kernel_type = kwargs["kernel_type"]
    switcher = {"linear": rkm.src.model.kernel.LinearKernel.LinearKernel,
                "rbf": rkm.src.model.kernel.RBFKernel.RBFKernel,
                "explicit": rkm.src.model.kernel.ExplicitNNKernel.ExplicitNNKernel,
                "implicit": rkm.src.model.kernel.ImplicitNNKernel.ImplicitNNKernel,
                "polynomial": rkm.src.model.kernel.PolynomialKernel.PolynomialKernel,
                "sigmoid": rkm.src.model.kernel.SigmoidKernel.SigmoidKernel,
                "indicator":  rkm.src.model.kernel.IndicatorKernel.IndicatorKernel,
                "hat": rkm.src.model.kernel.HatKernel.HatKernel,
                "nystrom": rkm.src.model.kernel.NystromKernel.NystromKernel}
    if kernel_type not in switcher:
        raise NameError("Invalid kernel type.")
    return switcher[kernel_type](**kwargs)

