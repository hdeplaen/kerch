==========
RBF Kernel
==========

Class
=====

.. autoclass:: kerch.kernel.RBF
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

    x = np.sin(np.arange(50) / np.pi)
    plt.figure(0)
    plt.plot(x)

    k = kerch.kernel.RBF(sample=x)

    plt.figure(1)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Sigma = "+str(k.sigma))

    k.sigma = 1

    plt.figure(2)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Sigma = "+str(k.sigma))

Time
----

.. plot::
    :include-source:

    import kerch
    from matplotlib import pyplot as plt

    k = kerch.kernel.RBF(sample=range(10), sigma=3)

    plt.imshow(k.K)
    plt.colorbar()
    plt.title("RBF with sigma " + str(k.sigma))


Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.RBF(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='rbf', **kwargs)

Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.RBF
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module