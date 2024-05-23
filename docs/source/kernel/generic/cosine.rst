=============
Cosine Kernel
=============

Class
=====


.. autoclass:: kerch.kernel.Cosine
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
    :show-inheritance:


Examples
========


Connection with Linear
----------------------
Essentially, a cosine kernel is the same as a linear kernel with normalization (as first transformation if multiple
are applied).

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.sin(np.arange(50) / np.pi)

    k_cos = kerch.kernel.Cosine(sample=x)
    k_lin = kerch.kernel.Linear(sample=x, kernel_transform=['normalize'])

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(k_cos.K)
    axs[0].set_title("Cosine")

    im = axs[1].imshow(k_lin.K)
    axs[1].set_title("Normalized Linear")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')

Multiple Dimensions
-------------------

The checkboard pattern appearing is a consequence of the normalization of one-dimensional input.
This does not happen in higher dimensions.

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    sin = np.expand_dims(np.sin(np.arange(50) / np.pi), axis=1)
    log = np.expand_dims(np.sin(np.log(np.arange(50)+1)), axis=1)

    x1 = sin
    x2 = np.concatenate((sin,log), axis=1)

    k1 = kerch.kernel.Cosine(sample=x1)
    k2 = kerch.kernel.Cosine(sample=x2)

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(k1.K)
    axs[0].set_title("One Dimension")

    im = axs[1].imshow(k2.K)
    axs[1].set_title("Two Dimensions")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')


Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.Cosine(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='cosine', **kwargs)


Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Cosine
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module