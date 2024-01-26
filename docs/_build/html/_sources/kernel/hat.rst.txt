==========
Hat Kernel
==========


Class
=====

.. autoclass:: kerch.kernel.Hat
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

    k = kerch.kernel.Hat(sample=range(10), lag=3)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Hat with lag " + str(k.lag))

Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.Hat(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='hat', **kwargs)


Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Hat
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module