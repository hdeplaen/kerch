import unittest
import kerch
from test import TestKPCA, TestKernels

if __name__ == '__main__':
    kerch.set_log_level(40)  # only print errors
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestKernels))
    suite.addTests(unittest.makeSuite(TestKPCA))
    unittest.TextTestRunner(verbosity=2).run(suite)
