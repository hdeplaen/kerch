# --- hide: start ---
import kerch
import torch

torch.manual_seed(0)

sample = torch.randn(5, 3)
oos = torch.randn(2, 3)

kpca_light_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                    dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                    sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                    kernel_transform=['center'],    # we want the kernel to be center
                                    cache_level='light')            # a 'light' cache level (only related to the sample)

kpca_heavy_cache = kerch.level.KPCA(sample=sample,                  # idem
                                    dim_output=2,                   # idem
                                    sample_transform=['min'],       # idem
                                    kernel_transform=['center'],    # idem
                                    cache_level='total')            # a 'total' cache level (logs everything)

# --- hide: stop ---

# kpca_light_cache.print_cache()
# kpca_heavy_cache.print_cache()

kpca_light_cache.solve()
kpca_heavy_cache.solve()

kpca_light_cache.forward(oos)
kpca_heavy_cache.forward(oos)

# kpca_light_cache.print_cache()
# kpca_heavy_cache.print_cache()


kpca_light_cache.reset()
kpca_heavy_cache.reset()

# kpca_light_cache.print_cache()
kpca_heavy_cache.print_cache()

