"""
Author: HENRI DE PLAEN
Date: May 2022
License: MIT
"""

import unittest
import torch
import kerch
kerch.set_log_level(40)  # only print errors
unittest.TestCase.__str__ = lambda x: ""


class TestKPCA(unittest.TestCase):
    r"""
    These tests verify the consistency of the computations of the kernels.
    """

    def __init__(self, *args, **kwargs):
        super(TestKPCA, self).__init__(*args, **kwargs)

        self.NUM_DATA = 10
        self.DIM_INPUT = 6
        self.DIM_FEATURE = 4
        self.DIM_OUTPUT = 2

        self.primal_types = ["linear", "nystrom"]
        self.dual_types = ["linear", "rbf"]

        self._gen_data()


    def _gen_data(self):
        size = self.NUM_DATA // 2

        g1 = torch.randn(1)[0] * torch.randn(size, self.DIM_INPUT) + torch.randn(1,self.DIM_INPUT).repeat(size, 1)
        g2 = torch.randn(1)[0] * torch.randn(size, self.DIM_INPUT) + torch.randn(1,self.DIM_INPUT).repeat(size, 1)

        self.x = torch.cat((g1, g2))

    def test_primal(self):
        """
        The primal principal component analysis can be computed and reconstructed."
        """
        for type in self.primal_types:
            mdl = kerch.rkm.KPCA(type=type,
                                 sample=self.x,
                                 representation="primal",
                                 dim_output=self.DIM_FEATURE)
            mdl.solve()
            var = mdl.relative_variance()
            self.assertLess(self.DIM_FEATURE / self.NUM_DATA, var)

    def test_dual(self):
        """
        The dual principal component analysis can be computed and reconstructed."
        """
        for type in self.dual_types:
            mdl = kerch.rkm.KPCA(type=type,
                                 sample=self.x,
                                 representation="dual",
                                 dim_output=self.DIM_FEATURE)
            mdl.solve()
            var = mdl.relative_variance()
            self.assertLess(self.DIM_FEATURE/self.NUM_DATA, var)

    def test_primal_train(self):
        """
        Optimizing leads to the same solution as performing eigendecomposition in primal.
        """
        for type in self.primal_types:
            mdl1 = kerch.rkm.KPCA(type=type,
                                  sample=self.x,
                                  representation="primal",
                                  dim_output=self.DIM_FEATURE)
            mdl1.fit(method="optimize", verbose=False, stiefel_lr=5.e-3, maxiter=500)
            var1 = mdl1.relative_variance()
            ##
            mdl2 = kerch.rkm.KPCA(type=type,
                                  sample=self.x,
                                  representation="primal",
                                  dim_output=self.DIM_FEATURE)
            mdl2.solve()
            var2 = mdl2.relative_variance()
            ##
            self.assertLess(var1, var2)
            self.assertAlmostEqual(var1, var2, places=1)

    def test_dual_train(self):
        """
        Optimizing leads to the same solution as performing eigendecomposition in dual.
        """
        for type in self.dual_types:
            mdl1 = kerch.rkm.KPCA(type=type,
                                  sample=self.x,
                                  representation="dual",
                                  dim_output=self.DIM_FEATURE)
            mdl1.fit(method="optimize", verbose=False, stiefel_lr=5.e-3, maxiter=500)
            var1 = mdl1.relative_variance()
            ##
            mdl2 = kerch.rkm.KPCA(type=type,
                                  sample=self.x,
                                  representation="dual",
                                  dim_output=self.DIM_FEATURE)
            mdl2.solve()
            var2 = mdl2.relative_variance()
            ##
            self.assertLess(var1, var2)
            self.assertAlmostEqual(var1, var2, places=1)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKPCA)
    unittest.TextTestRunner(verbosity=2).run(suite)
