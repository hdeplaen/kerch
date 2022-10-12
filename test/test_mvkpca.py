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
