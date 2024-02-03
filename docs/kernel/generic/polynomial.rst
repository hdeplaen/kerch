=================
Polynomial Kernel
=================

Class
=====

.. autoclass:: kerch.kernel.Polynomial
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

    k = kerch.kernel.Polynomial(sample=x)
    plt.figure(1)
    plt.imshow(k.K)
    plt.title(f"Alpha = {k.alpha}, Beta = {k.beta}")
    plt.colorbar()

Factory
-------

The following lines are equivalent:

.. code-block::

    k = kerch.kernel.Polynomial(**kwargs)


.. code-block::

    k = kerch.kernel.factory(kernel_type='polynomial', **kwargs)


Influence of the parameters
---------------------------

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.sin(np.arange(50) / np.pi)

    k1 = kerch.kernel.Polynomial(sample=x, alpha=2, beta=1)
    k2 = kerch.kernel.Polynomial(sample=x, alpha=2, beta=5)
    k3 = kerch.kernel.Polynomial(sample=x, alpha=5, beta=1)
    k4 = kerch.kernel.Polynomial(sample=x, alpha=5, beta=5)

    fig, axs = plt.subplots(2, 2)

    axs[0,0].imshow(k1.K)
    axs[0,0].set_title(f"Alpha = {k1.alpha}, Beta = {k1.beta}")

    axs[0,1].imshow(k2.K)
    axs[0,1].set_title(f"Alpha = {k2.alpha}, Beta = {k2.beta}")

    axs[1,0].imshow(k3.K)
    axs[1,0].set_title(f"Alpha = {k3.alpha}, Beta = {k3.beta}")

    im = axs[1,1].imshow(k4.K)
    axs[1,1].set_title(f"Alpha = {k4.alpha}, Beta = {k4.beta}")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')


In the following example, the kernels are also to ease the readability of the effects.
The diagonal always becomes the unity when normalizing, hence the more pronounced difference with the non-normalized example above.

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.sin(np.arange(50) / np.pi)

    k1 = kerch.kernel.Polynomial(sample=x, alpha=2, beta=1, kernel_transform=['normalize'])
    k2 = kerch.kernel.Polynomial(sample=x, alpha=2, beta=5, kernel_transform=['normalize'])
    k3 = kerch.kernel.Polynomial(sample=x, alpha=5, beta=1, kernel_transform=['normalize'])
    k4 = kerch.kernel.Polynomial(sample=x, alpha=5, beta=5, kernel_transform=['normalize'])

    fig, axs = plt.subplots(2, 2)

    axs[0,0].imshow(k1.K)
    axs[0,0].set_title(f"Alpha = {k1.alpha}, Beta = {k1.beta}")

    axs[0,1].imshow(k2.K)
    axs[0,1].set_title(f"Alpha = {k2.alpha}, Beta = {k2.beta}")

    axs[1,0].imshow(k3.K)
    axs[1,0].set_title(f"Alpha = {k3.alpha}, Beta = {k3.beta}")

    im = axs[1,1].imshow(k4.K)
    axs[1,1].set_title(f"Alpha = {k4.alpha}, Beta = {k4.beta}")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')


Explicit and Implicit
---------------------

The polynomial kernel can have its kernel matrix computed through the explicit feature map as :math:`k(x,y) = \phi(x)^\top\phi(y)`
and implicitly using :math:`k(x,y) = \left(x^\top y + \beta\right)^\alpha`. The following confirms that both are equivalent.


.. plot::
    :include-source:

    import kerch
    import torch
    from matplotlib import pyplot as plt

    num, dim_input = 10,3

    x = torch.randn(num, dim_input)
    oos = torch.randn(num, dim_input)

    k = kerch.kernel.Polynomial(sample=x, alpha=3, kernel_transform=['center', 'normalize'])

    fig, axs = plt.subplots(2, 2)

    axs[0,0].imshow(k.k(explicit=True))
    axs[0,0].set_title("Explicit (sample)")

    axs[0,1].imshow(k.k(explicit=False))
    axs[0,1].set_title("Implicit (sample)")

    axs[1,0].imshow(k.k(x=oos, y=oos, explicit=True))
    axs[1,0].set_title("Implicit (out-of-sample)")

    im = axs[1,1].imshow(k.k(x=oos, y=oos, explicit=False))
    axs[1,1].set_title("Explicit (out-of-sample)")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')


Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Polynomial
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module