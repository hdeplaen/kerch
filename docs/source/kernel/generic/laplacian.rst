================
Laplacian Kernel
================

.. autoclass:: kerch.kernel.Laplacian
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

    k = kerch.kernel.Laplacian(sample=x)

    plt.figure(1)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title(f"Sigma = {k.sigma:.2f}")

    k.sigma = 2

    plt.figure(2)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title(f"Sigma = {k.sigma:.2f}")

Comparison with RBF
-------------------
As the 2-norm between the inputs is not squared, the result is essentially more drastically descreasing in the bulk,
but heavier in the tail compared to the :class:`~kerch.kernel.RBF` kernel. This will also lead to a proportionally
higher sigma for a "similar" result.

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.sin(np.arange(50) / np.pi)

    # automatic bandwidth with heuristic
    k_laplacian = kerch.kernel.Laplacian(sample=x)
    k_rbf = kerch.kernel.RBF(sample=x)

    fig1, axs1 = plt.subplots(1, 2)

    axs1[0].imshow(k_laplacian.K)
    axs1[0].set_title(f"Laplacian ($\sigma$={k_laplacian.sigma:.2f})")

    im1 = axs1[1].imshow(k_rbf.K)
    axs1[1].set_title(f"RBF ($\sigma$={k_rbf.sigma:.2f})")

    fig1.colorbar(im1, ax=axs1.ravel().tolist(), orientation='horizontal')

    #  unity bandwidth
    k_laplacian_sigma1 = kerch.kernel.Laplacian(sample=x, sigma=1)
    f_rbf_sigma1 = kerch.kernel.RBF(sample=x, sigma=1)

    fig2, axs2 = plt.subplots(1, 2)

    axs2[0].imshow(k_laplacian_sigma1.K)
    axs2[0].set_title(f"Laplacian ($\sigma$={k_laplacian_sigma1.sigma})")

    im2 = axs2[1].imshow(f_rbf_sigma1.K)
    axs2[1].set_title(f"RBF ($\sigma$={f_rbf_sigma1.sigma})")

    fig2.colorbar(im2, ax=axs2.ravel().tolist(), orientation='horizontal')


.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    x = torch.linspace(-5, 5, 200)
    k_rbf = kerch.kernel.RBF(sample=x, sigma=1)
    k_laplacian = kerch.kernel.Laplacian(sample=x, sigma=1)
    shape = torch.cat((k_rbf.k(y=0), k_laplacian.k(y=0)), dim=1)

    plt.figure()
    plt.plot(x, shape)
    plt.title('Kernel Shape')
    plt.legend(['RBF',
                'Laplacian'])
    plt.xlabel('x')
    plt.ylabel('k(x,y=0)')


Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.Laplacian(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='laplacian', **kwargs)


Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Laplacian
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module