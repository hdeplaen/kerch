================
Indicator Kernel
================

Class
=====

.. autoclass:: kerch.kernel.Indicator
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

    k = kerch.kernel.Indicator(sample=range(10), lag=3)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Indicator with lag " + str(k.lag))

Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.Indicator(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='indicator', **kwargs)



Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Indicator
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module