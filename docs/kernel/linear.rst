=============
Linear Kernel
=============

Class
=====

.. autoclass:: kerch.kernel.linear
   :members:
   :inherited-members: Module
   :undoc-members:
   :exclude-members: training, dump_patches
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

    k = kerch.kernel.linear(sample=x)
    plt.figure(1)
    plt.imshow(k.K)
    plt.colorbar()
