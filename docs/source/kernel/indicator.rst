================
Indicator Kernel
================

Class
=====

.. autoclass:: kerch.kernel.indicator
   :members:
   :inherited-members: Module
   :undoc-members:
   :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
   :show-inheritance:


Examples
========

Linear (Time)
-------------

.. plot::
    :include-source:

    import kerch
    from matplotlib import pyplot as plt

    k = kerch.kernel.indicator(sample=range(10), lag=3)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Indicator with lag " + str(k.lag))
