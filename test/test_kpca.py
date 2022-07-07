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

        # DATASET
        # self.source = 6 * torch.rand(self.NUM_INPUT) - 3
        # self.x = torch.stack((torch.sin(self.source), self.source / 6), dim=1)
        # self.y = torch.sum(self.x ** 2, dim=1)

        self.x = torch.rand((self.NUM_DATA, self.DIM_INPUT))
        self.y = torch.rand((self.NUM_DATA, self.DIM_OUTPUT))

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
            recon = mdl.reconstruct(self.x)
            diff = (recon - mdl.Phi)
            self.assertLess(torch.norm(diff).detach().numpy(), self.NUM_DATA / self.DIM_FEATURE, 0)

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
            recon = mdl.reconstruct(self.x)
            diff = (recon - mdl.K)
            self.assertLess(torch.norm(diff).detach().numpy(), self.NUM_DATA / self.DIM_FEATURE, 0)

    def test_primal_train(self):
        """
        Optimizing leads to the same solution as performing eigendecomposition in primal.
        """
        for type in self.primal_types:
            mdl1 = kerch.rkm.KPCA(type=type,
                                 sample=self.x,
                                 representation="primal",
                                 dim_output=self.DIM_FEATURE)
            mdl1.fit(method="optimize", verbose=False, lr=5.e-2, maxiter=50)
            var1 = mdl1.relative_variance()
            ##
            mdl2 = kerch.rkm.KPCA(type=type,
                                  sample=self.x,
                                  representation="primal",
                                  dim_output=self.DIM_FEATURE)
            mdl2.solve()
            var2 = mdl2.relative_variance()
            ##
            # self.assertLess(var1, var2)
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
            mdl1.fit(method="optimize", verbose=False, lr=5.e-2, maxiter=50)
            var1 = mdl1.relative_variance()
            ##
            mdl2 = kerch.rkm.KPCA(type=type,
                                  sample=self.x,
                                  representation="dual",
                                  dim_output=self.DIM_FEATURE)
            mdl2.solve()
            var2 = mdl2.relative_variance()
            ##
            # self.assertLess(var1, var2)
            self.assertAlmostEqual(var1, var2, places=1)

    def test_primal_mv(self):
        """
        Check Multi-View consistency in primal.
        """
        for type in self.primal_types:
            mdl = kerch.rkm.MVKPCA({"name": "x",
                                    "type": type,
                                    "sample": self.x},
                                   {"name": "y",
                                    "type": type,
                                    "sample": self.y},
                                   representation="primal")
            mdl.solve()
            recon_x = mdl.reconstruct('x')
            recon_y = mdl.reconstruct('y')
            phi_x = mdl.phi('x')
            phi_y = mdl.phi('y')
            self.assertAlmostEqual(recon_x.sum().detach().cpu().numpy(),
                                   phi_x.sum().detach().cpu().numpy())
            self.assertAlmostEqual(recon_y.sum().detach().cpu().numpy(),
                                   phi_y.sum().detach().cpu().numpy())

    def test_dual_mv(self):
        """
        Check Multi-View consistency in dual.
        """
        pass
        # for type in self.dual_types:
        #     mdl = kerch.rkm.MVKPCA({"name": "x",
        #                             "type": type,
        #                             "sample": self.x},
        #                            {"name": "y",
        #                             "type": type,
        #                             "sample": self.y},
        #                            representation="dual")
        #     mdl.solve()
        #     recon_x = mdl.reconstruct('x')
        #     recon_y = mdl.reconstruct('y')
        #     k_x = mdl.k('x')
        #     k_y = mdl.k('y')
        #     self.assertAlmostEqual(recon_x.sum().detach().cpu().numpy(),
        #                            k_x.sum().detach().cpu().numpy())
        #     self.assertAlmostEqual(recon_y.sum().detach().cpu().numpy(),
        #                            k_y.sum().detach().cpu().numpy())

    def test_primal_mv_train(self):
        """
        Tests optimization of the multi-view KPCA in primal formulation.
        """
        for type in self.primal_types:
            mdl1 = kerch.rkm.MVKPCA({"name": "x",
                                     "type": type,
                                     "sample": self.x},
                                    {"name": "y",
                                     "type": type,
                                     "sample": self.y},
                                    representation="primal",
                                    dim_output=self.DIM_FEATURE)
            mdl1.fit(method="optimize", verbose=False, lr=5.e-2, maxiter=50)
            ##
            var1 = mdl1.relative_variance()
            ##
            mdl2 = kerch.rkm.MVKPCA({"name": "x",
                                     "type": type,
                                     "sample": self.x},
                                    {"name": "y",
                                     "type": type,
                                     "sample": self.y},
                                    representation="primal",
                                    dim_output=self.DIM_FEATURE)
            mdl2.fit(method="exact")
            var2 = mdl2.relative_variance()
            ##
            # self.assertLess(var1, var2)
            self.assertAlmostEqual(var1, var2, places=1)

    def test_dual_mv_train(self):
        """
        Tests optimization of the multi-view KPCA in dual formulation.
        """
        for type in self.dual_types:
            mdl1 = kerch.rkm.MVKPCA({"name": "x",
                                     "type": type,
                                     "sample": self.x},
                                    {"name": "y",
                                     "type": type,
                                     "sample": self.y},
                                    representation="dual",
                                    dim_output=self.DIM_FEATURE)
            mdl1.fit(method="optimize", verbose=False, lr=5.e-2, maxiter=50)
            ##
            var1 = mdl1.relative_variance()
            ##
            mdl2 = kerch.rkm.MVKPCA({"name": "x",
                                     "type": type,
                                     "sample": self.x},
                                    {"name": "y",
                                     "type": type,
                                     "sample": self.y},
                                    representation="dual",
                                    dim_output=self.DIM_FEATURE)
            mdl2.fit(method="exact")
            var2 = mdl2.relative_variance()
            ##
            # self.assertLess(var1, var2)
            self.assertAlmostEqual(var1, var2, places=1)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKPCA)
    unittest.TextTestRunner(verbosity=2).run(suite)
