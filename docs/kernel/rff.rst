==============================
Random Fourier Features Kernel
==============================

Class
=====

.. autoclass:: kerch.kernel.rff
   :members:
   :inherited-members: Module
   :undoc-members:
   :exclude-members: training, dump_patches, sample_as_param
   :show-inheritance:

Examples
========

Sine
----

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.sin(np.arange(50) / np.pi)

    k_rbf = kerch.kernel.rbf(sample=x, sigma=1)
    k_rff = kerch.kernel.rff(sample=x, num_weights=50)

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(k_rbf.K)
    axs[0].set_title("RBF")

    im = axs[1].imshow(k_rff.K)
    axs[1].set_title("RFF")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')