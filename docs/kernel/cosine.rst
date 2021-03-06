=============
Cosine Kernel
=============

Class
=====


.. autoclass:: kerch.kernel.cosine
   :members:
   :inherited-members: Module
   :undoc-members:
   :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
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

    x = np.sin(np.arange(50) / np.pi) + 1.5
    plt.figure(0)
    plt.plot(x)

    k = kerch.kernel.cosine(sample=x, center=True)
    plt.figure(1)
    plt.imshow(k.K)
    plt.colorbar()
