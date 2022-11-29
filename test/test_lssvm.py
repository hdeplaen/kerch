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

#TODO: stochastic learning not working. check why.

class TestLSSVM(unittest.TestCase):
    r"""
    These tests verify the consistency of the computations of the kernels.
    """

    def __init__(self, *args, **kwargs):
        super(TestLSSVM, self).__init__(*args, **kwargs)

        self.NUM_DATA = 100

        self.primal_types = ["linear", "nystrom"]
        self.dual_types = ["linear", "rbf"]

        self._gen_data()


    def _gen_data(self):
        from sklearn import datasets
        self.x, self.y = datasets.make_moons(self.NUM_DATA, noise=.1)
        self.DIM_INPUT = self.x.shape[1]
        self.DIM_OUTPUT = 1

    def test_primal(self):
        """
        The primal principal component analysis can be computed and reconstructed."
        """
        for type in self.primal_types:
            mdl = kerch.rkm.LSSVM(type=type,
                                  sample=self.x,
                                  targets=self.y,
                                  representation="primal")
            mdl.solve()
            kerch.plot.classifier_level(mdl)

    def test_dual(self):
        """
        The dual principal component analysis can be computed and reconstructed."
        """
        for type in self.dual_types:
            mdl = kerch.rkm.LSSVM(type=type,
                                  sample=self.x,
                                  targets=self.y,
                                  representation="dual")
            mdl.solve()
            recon = mdl.forward(self.x)
            kerch.plot.classifier_level(mdl)

    def test_primal_train(self):
        """
        Optimizing leads to the same solution as performing eigendecomposition in primal.
        """
        for type in self.primal_types:
            mdl1 = kerch.rkm.LSSVM(type=type,
                                   sample=self.x,
                                   targets=self.y,
                                   representation="primal")
            mdl1.fit(method="optimize", verbose=False, euclidean_lr=1.e-3, maxiter=100)
            kerch.plot.classifier_level(mdl1)
            # ##
            # mdl2 = kerch.rkm.LSSVM(type=type,
            #                       sample=self.x,
            #                       representation="primal")
            # mdl2.solve()
            # var2 = mdl2.model_variance()
            # ##
            # self.assertLess(var1, var2)
            # self.assertAlmostEqual(var1, var2, places=0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKPCA)
    unittest.TextTestRunner(verbosity=2).run(suite)
