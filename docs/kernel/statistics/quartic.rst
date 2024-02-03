=========================
Quartic (biweight) Kernel
=========================


Class
=====

.. autoclass:: kerch.kernel.Quartic
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

    x = torch.linspace(-1.2, 1.2, 100)
    k = kerch.kernel.Quartic(sample=x, sigma=1)
    shape = k.k(y=0).squeeze()

    plt.figure()
    plt.plot(x, shape)
    plt.title('Quartic Shape')
    plt.xlabel('x')
    plt.ylabel('k(x,y=0)')


Random
------

.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    num_input, dim_input = 20, 3
    sample = torch.randn(num_input, dim_input)

    k1 = kerch.kernel.Quartic(sample=sample, sigma=1.5)
    k2 = kerch.kernel.Quartic(sample=sample, distance='chebyshev', sigma=1.5)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(k1.K)
    axs[0].set_title("Quartic (Euclidean)")
    im = axs[1].imshow(k2.K)
    axs[1].set_title("Quartic (Chebyshev)")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Quartic
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module
