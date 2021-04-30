import rkm.model.kernel.ExplicitKernel as ExplicitKernel
import rkm.model.kernel.ImplicitKernel as ImplicitKernel
import rkm.model.kernel.LinearKernel as LinearKernel
import rkm.model.kernel.PolynomialKernel as PolynomialKernel
import rkm.model.kernel.RBFKernel as RBFKernel
import rkm.model.kernel.SigmoidKernel as SigmoidKernel

class KernelFactory:
    @staticmethod
    def create_kernel(kernel_type, **kwargs):
        switcher = {"linear": LinearKernel.LinearKernel,
                    "rbf": RBFKernel.RBFKernel,
                    "explicit": ExplicitKernel.ExplicitKernel,
                    "implicit": ImplicitKernel.ImplicitKernel,
                    "polynomial": PolynomialKernel.PolynomialKernel,
                    "sigmoid": SigmoidKernel.SigmoidKernel}
        if kernel_type not in switcher:
            raise NameError("Invalid kernel type.")
        return switcher[kernel_type](**kwargs)

