"""
Various sanity tests

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: September 2021
"""

import unittest

from rkm.expes.data import data
import rkm.model.rkm as rkm

class TestLevels(unittest.TestCase):
    def _test_prototype(self, type, representation, kernel, classifier=False):
        # DATASET
        train, _, _, info = data.factory("two_moons", 50)
        x, y = train
        kernel_params = {"kernel_type": kernel}
        level_params = {"size_in": 2,
                        "size_out": 1,
                        "representation": representation,
                        "classifier": classifier,
                        "kernel": kernel_params}
        learn_params = {"type": "sgd",
                        "init": False,
                        "maxiter": 1.e+4,
                        "lr": 0.02,
                        "tol": 1.e-9}


        # HARD
        hard = rkm.RKM()
        hard_params = {**level_params, "constraint": "hard"}
        hard.append_level(type=type, **hard_params)
        out_hard = hard.learn(x, y, **learn_params)

        # SOFT
        soft = rkm.RKM()
        soft_params = {**level_params, "constraint": "soft"}
        soft.append_level(type=type, **soft_params)
        out_soft = soft.learn(x, y, **learn_params)

        # TEST
        self.assertEqual(out_hard, out_soft)

    def test_primal_kpca(self):
        self._test_prototype(type="kpca", representation="primal", kernel="linear")
        self._test_prototype(type="kpca", representation="primal", kernel="polynomial")

    def test_dual_kpca(self):
        self._test_prototype(type="kpca", representation="dual", kernel="rbf")
        self._test_prototype(type="kpca", representation="dual", kernel="polynomial")
        self._test_prototype(type="kpca", representation="dual", kernel="sigmoid")

    def test_primal_lssvm(self):
        self._test_prototype(type="lssvm", representation="primal", kernel="linear")
        self._test_prototype(type="lssvm", representation="primal", kernel="polynomial")

    def test_dual_lssvm(self):
        self._test_prototype(type="lssvm", representation="dual", kernel="rbf")
        self._test_prototype(type="lssvm", representation="dual", kernel="polynomial")
        self._test_prototype(type="lssvm", representation="dual", kernel="sigmoid")

    def test_primal_lssvm_classifier(self):
        self._test_prototype(type="lssvm", representation="primal", kernel="linear", classifier=True)
        self._test_prototype(type="lssvm", representation="primal", kernel="polynomial", classifier=True)

    def test_dual_lssvm_classifier(self):
        self._test_prototype(type="lssvm", representation="dual", kernel="rbf", classifier=True)
        self._test_prototype(type="lssvm", representation="dual", kernel="polynomial", classifier=True)
        self._test_prototype(type="lssvm", representation="dual", kernel="sigmoid", classifier=True)

class Suites():
    @staticmethod
    def levels_suite():
        suite = unittest.TestSuite()
        suite.addTest(TestLevels('test_primal_kpca'))
        suite.addTest(TestLevels('test_dual_kpca'))
        suite.addTest(TestLevels('test_primal_lssvm'))
        suite.addTest(TestLevels('test_dual_lssvm'))
        suite.addTest(TestLevels('test_primal_lssvm_classifier'))
        suite.addTest(TestLevels('test_dual_lssvm_classifier'))
        return suite