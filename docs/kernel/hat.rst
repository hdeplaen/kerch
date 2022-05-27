==========
Hat Kernel
==========


Class
=====

.. autoclass:: rkm.kernel.hat
   :members:
   :inherited-members: Module
   :undoc-members:
   :exclude-members: training, dump_patches, phi_sample, phi, C
   :show-inheritance:


Examples
========

Linear (Time)
-------------

.. plot::
    :include-source:

    import rkm
    from matplotlib import pyplot as plt

    k = rkm.kernel.hat(sample=range(10), lag=3)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Hat with lag " + str(k.lag))
