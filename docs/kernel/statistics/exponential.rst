==================
Exponential Kernel
==================


Class
=====

.. autoclass:: kerch.kernel.Exponential
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
    :show-inheritance:


Example
=======

Shape
-----

.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    x = torch.linspace(-5, 5, 200)
    k_squared = kerch.kernel.Exponential(sample=x, sigma=1)                     # same as RBF kernel
    k_non_squared = kerch.kernel.Exponential(sample=x, sigma=1, squared=False)  # same as Laplacian kernel
    shape = torch.cat((k_squared.k(y=0), k_non_squared.k(y=0)), dim=1)

    plt.figure()
    plt.plot(x, shape)
    plt.title('Exponential Shape')
    plt.legend(['Squared (default)', 'Non-Squared'])
    plt.xlabel('x')
    plt.ylabel('k(x,y=0)')


Random
------

.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    num_input, dim_input = 20, 5
    sample = torch.randn(num_input, dim_input)

    k1 = kerch.kernel.Exponential(sample=sample, sigma=3)
    k2 = kerch.kernel.Exponential(sample=sample, distance='chebyshev', sigma=3)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(k1.K)
    axs[0].set_title("Exponential (Euclidean)")
    im = axs[1].imshow(k2.K)
    axs[1].set_title("Exponential (Chebyshev)")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Exponential
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module
