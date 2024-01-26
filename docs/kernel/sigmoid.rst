==============
Sigmoid Kernel
==============

Class
=====

.. autoclass:: kerch.kernel.Sigmoid
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
    :show-inheritance:

Example
=======

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

    k = kerch.kernel.Sigmoid(sample=x)
    plt.figure(1)
    plt.imshow(k.K)
    plt.colorbar()


Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.Sigmoid(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='sigmoid', **kwargs)


Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Sigmoid
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module
