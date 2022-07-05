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

class TestKernels(unittest.TestCase):
    r"""
    These tests verify the consistency of the computations of the kernels.
    """
    def __init__(self, *args, **kwargs):
        super(TestKernels, self).__init__(*args, **kwargs)

        # some tests require psd matrices
        self.tested_kernels = ['linear',
                               'rbf',
                               'laplacian',
                               'cosine',
                               'hat',
                               'indicator',
                               'polynomial',
                               'additive_chi2',
                               'skewed_chi2']

        self.sample = range(1,5)

    def test_kernel_matrices(self):
        """
        Verifies if the sample kernel matrices can be computed."
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample)
            self.assertIsInstance(k.K, torch.Tensor, msg=type_name)

    def test_symmetry(self):
        """
        Verifies if the sample kernel matrices are symmetric."
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample)
            self.assertAlmostEqual(torch.norm(k.K - k.K.T, p='fro').numpy(), 0, msg=type_name)

    def test_psd(self):
        """
        Verifies if the sample kernel matrices are positive semi-definie."
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample)
            e = torch.linalg.eigvals(k.K).real < -1.e-15
            self.assertEqual(e.sum().numpy(), 0, msg=self.sample)

    def test_centered_kernel_matrices(self):
        """
        Verifies if the centered kernel matrices have zero sum.
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample, center=True)
            self.assertAlmostEqual(k.K.sum().numpy(), 0, msg=type_name)

    def test_normalized_kernel_matrices(self):
        """
        Verifies if the normalized kernel matrices have unit diagonal (or negative).
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample, normalize=True)
            self.assertAlmostEqual(torch.abs(torch.diag(k.K)).sum().numpy(), len(self.sample), msg=type_name)

    def test_centered_normalized_kernel_matrices(self):
        """
        Verifies if the normalized after centered kernel matrices have unit diagonal (or negative).
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample, center=True, normalize=True)
            self.assertAlmostEqual(torch.diag(k.K).sum().numpy(), len(self.sample), msg=type_name)

    def test_out_of_sample(self):
        """
        Verifies if the out-of-sample kernel matrices correspond to the sample one.
        """
        for type_name in self.tested_kernels:
            sample = self.sample
            k = kerch.kernel.factory(type=type_name, sample=sample)
            self.assertAlmostEqual(torch.norm(k.K - k.k(x=sample, y=sample), p='fro').numpy(), 0, msg=type_name)

    def test_out_of_sample_centered(self):
        """
        Verifies if the out-of-sample centered kernel matrices correspond to the sample one.
        """
        for type_name in self.tested_kernels:
            sample = self.sample
            k = kerch.kernel.factory(type=type_name, sample=sample, center=True)
            self.assertAlmostEqual(torch.norm(k.K - k.k(x=sample, y=sample), p='fro').numpy(), 0, msg=type_name)

    def test_out_of_sample_normalized(self):
        """
        Verifies if the out-of-sample normalized kernel matrices correspond to the sample one.
        """
        for type_name in self.tested_kernels:
            sample = self.sample
            k = kerch.kernel.factory(type=type_name, sample=sample, normalize=True)
            self.assertAlmostEqual(torch.norm(k.K - k.k(x=sample, y=sample), p='fro').numpy(), 0, msg=type_name)

    def test_out_of_sample_centered_normalized(self):
        """
        Verifies if the out-of-sample normalized after centered kernel matrices correspond to the sample one.
        """
        for type_name in self.tested_kernels:
            sample = self.sample
            k = kerch.kernel.factory(type=type_name, sample=sample, center=True, normalize=True)
            self.assertAlmostEqual(torch.norm(k.K - k.k(x=sample, y=sample), p='fro').numpy(), 0, msg=type_name)

    def test_nystrom_scratch(self):
        """
        Verifies the consistency of a Nyström kernel created from scratch.
        """
        sample = self.sample
        k_nystrom = kerch.kernel.nystrom(sample=sample)
        k_base = k_nystrom.base_kernel
        self.assertAlmostEqual(torch.norm(k_nystrom.K - k_base.K, p='fro').numpy(), 0)

    def test_nystrom_base(self):
        """
        Verifies the consistency of a Nyström kernel based on an existing kernel.
        """
        sample = self.sample
        k_base = kerch.kernel.rbf(sample=sample)
        k_nystrom = kerch.kernel.nystrom(base_kernel=k_base)
        self.assertAlmostEqual(torch.norm(k_nystrom.k() - k_base.K, p='fro').numpy(), 0)

    @unittest.skipUnless(kerch.gpu_available(), 'CUDA is not available for PyTorch on this machine.')
    def test_gpu(self):
        """
        Verifies if the kernels run on GPU.
        """
        for type_name in self.tested_kernels:
            k = kerch.kernel.factory(type=type_name, sample=self.sample)
            k = k.to(device='cuda')
            self.assertIsInstance(k.K, torch.Tensor, msg=type_name)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernels)
    unittest.TextTestRunner(verbosity=2).run(suite)
