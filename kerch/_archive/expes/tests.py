"""
Various sanity tests

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: September 2021
"""

import unittest
import numpy as np

from kerch._archive.expes.data import data
from kerch._archive.model import rkm as rkm


class TestLevels(unittest.TestCase):
    def _test_prototype(self, type, representation, kernel, classifier=False, lr=0.001):
        # DATASET
        tol = 1.e-2
        cuda = True
        plot = False
        verb = False
        num_data = 20
        train, _, _, info = data.factory("two_moons", num_data)
        x, y = train
        kernel_params = {"kernel_type": kernel}
        level_params = {"size_in": 2,
                        "size_out": 1,
                        "representation": representation,
                        "classifier": classifier,
                        "kernel": kernel_params,
                        "init_kernels": num_data}
        learn_params = {"type": "sgd",
                        "init": False,
                        "plot": False,
                        "maxiter": 5e+4,
                        "lr": lr,
                        "tol": 1.e-12}

        # HARD
        hard = rkm.RKM(cuda=cuda, verbose=verb, name='test')
        hard_params = {**level_params, "constraint": "hard"}
        hard.append_level(type=type, **hard_params)
        out_hard = hard.learn(x, y, **learn_params)

        # SOFT
        soft = rkm.RKM(cuda=cuda, verbose=verb)
        soft_params = {**level_params, "constraint": "soft"}
        soft.append_level(type=type, **soft_params)
        out_soft = soft.learn(x, y, **learn_params)

        # TEST
        if type == 'KPCA':
            out_soft = np.abs(out_soft)
            out_hard = np.abs(out_hard)
        error = np.mean(out_soft - out_hard)

        self.assertTrue(np.abs(error) <= tol)

    def test_primal_kpca(self):
        # print('PRIMAL KPCA')
        self._test_prototype(type="KPCA", representation="primal", kernel="linear")
        # self._test_prototype(type="KPCA", representation="primal", kernel="polynomial") # NOT IMPLEMENTED

    def test_dual_kpca(self):
        # print('DUAL KPCA')
        self._test_prototype(type="KPCA", representation="dual", kernel="linear", lr=0.001)
        self._test_prototype(type="KPCA", representation="dual", kernel="rbf", lr=0.01)
        # self._test_prototype(type="KPCA", representation="dual", kernel="polynomial") # ERROR
        # self._test_prototype(type="KPCA", representation="dual", kernel="sigmoid", lr=0.001)

    def test_primal_lssvm(self):
        # print('PRIMAL LSSVM')
        self._test_prototype(type="LSSVM", representation="primal", kernel="linear", lr=0.05)
        # self._test_prototype(type="LSSVM", representation="primal", kernel="polynomial")

    def test_dual_lssvm(self):
        # print('DUAL LSSVM')
        self._test_prototype(type="LSSVM", representation="dual", kernel="linear", lr=0.001)
        self._test_prototype(type="LSSVM", representation="dual", kernel="rbf", lr=0.1)
        # self._test_prototype(type="LSSVM", representation="dual", kernel="polynomial")
        # self._test_prototype(type="LSSVM", representation="dual", kernel="sigmoid", lr=0.05)

    def test_primal_lssvm_classifier(self):
        # print('PRIMAL LSSVM CLASSIFIER')
        self._test_prototype(type="LSSVM", representation="primal", kernel="linear", classifier=True, lr=0.01)
        # self._test_prototype(type="LSSVM", representation="primal", kernel="polynomial", classifier=True)

    def test_dual_lssvm_classifier(self):
        # print('DUAL LSSVM CLASSIFIER')
        self._test_prototype(type="LSSVM", representation="dual", kernel="linear", classifier=True, lr=0.001)
        self._test_prototype(type="LSSVM", representation="dual", kernel="rbf", classifier=True, lr=0.1)
        # self._test_prototype(type="LSSVM", representation="dual", kernel="polynomial", classifier=True)
        # self._test_prototype(type="LSSVM", representation="dual", kernel="sigmoid", classifier=True, lr=0.002)

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