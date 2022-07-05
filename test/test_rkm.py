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


class TestRKM(unittest.TestCase):
    r"""
    These tests verify the consistency of the computations of the kernels.
    """

    def __init__(self, *args, **kwargs):
        super(TestRKM, self).__init__(*args, **kwargs)

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

    def test_primal_kpca(self):
        """
        Verifies if a primal principal component analysis can be computed and reconstructed."
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

    def test_dual_kpca(self):
        """
        Verifies if a dual principal component analysis can be computed and reconstructed."
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
        for type in self.primal_types:
            mdl = kerch.rkm.KPCA(type=type,
                                 sample=self.x,
                                 representation="primal",
                                 dim_output=self.DIM_FEATURE)
            mdl.fit(method="optimize", verbose=True)
            pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRKM)
    unittest.TextTestRunner(verbosity=2).run(suite)
