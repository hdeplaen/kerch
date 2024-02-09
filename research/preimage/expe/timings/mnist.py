import kerch
import torch.utils.data
import torchvision
import time
import numpy as np
import logging
from tqdm import tqdm

torch.manual_seed(95)
kerch.set_logging_level(logging.ERROR)

# preliminaries
kernel_type = ["rff", "rf_lrelu", "rf_lrelu", "rf_lrelu", "rf_arcsin"]
alpha = [0, 0, .1, 1, 0]
num_weights = [10, 20, 50, 100, 200, 500, 1000]
num_components = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
num_data = 1000
n_iter = 10

timings_train = np.zeros((len(kernel_type), len(num_weights), len(num_components), n_iter))
timings_recon = np.zeros((len(kernel_type), len(num_weights), len(num_components), n_iter))
rel_var = np.zeros((len(kernel_type), len(num_weights), len(num_components), n_iter))


# dataset
mnist = torchvision.datasets.MNIST('.../files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()]))
sampler = torch.utils.data.RandomSampler(mnist, replacement=True)
loader = torch.utils.data.DataLoader(mnist, batch_size=num_data, sampler=sampler)

for l in tqdm(range(n_iter), desc='iter', position=0):
    sample, _ = next(iter(loader))
    sample = sample.view(sample.shape[0], -1)
    for i, (kt, a) in enumerate(zip(kernel_type, alpha)):
        for j, nw in enumerate(num_weights):
            for k, nc in enumerate(num_components):
                # training
                t0 = time.time()
                kpca = kerch.level.KPCA(kernel_type=kt,
                                        num_weights=nw,
                                        sample=sample,
                                        dim_output=nc,
                                        sample_transform=['standardize'],
                                        kernel_transform=['center'],
                                        representation='primal',
                                        alpha=a)
                kpca.solve()
                timings_train[i, j, k, l] = time.time() - t0

                # variance
                rel_var[i, j, k, l] = kpca.relative_variance()

                # reconstruction
                t0 = time.time()
                try:
                    sample_recon = kpca.explicit_preimage(kpca.Phi)
                    timings_recon[i, j, k, l] = time.time() - t0
                except kerch.utils.BijectionError:
                    pass

                # save
                np.savez('mnist', timings_train, timings_recon, rel_var, num_components, num_weights, alpha, kernel_type)
