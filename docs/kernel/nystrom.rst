==============
Nyström Kernel
==============

This kernel allows to approximate explicit feature maps for kernels whose explicit representation lives in an
infinite-dimensional Hilbert space. Through eigendecomposition, we are able to find
a base in the RKHS spanned by the provided sample. By looking at the RKHS coefficients for each datapoint, we are able to
recover an approximate explicit feature map.

.. autoclass:: kerch.kernel.Nystrom
   :members:
   :inherited-members: Module
   :undoc-members:
   :exclude-members: training, dump_patches
   :show-inheritance:


Example
=======

Explicit RBF
------------

We first consider an RBF kernel and will approximate its explicit feature map both for the sample input and
an out-of-sample input.

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    sample = np.sin(np.arange(0,15) / np.pi) + .1
    oos = np.sin(np.arange(15,30) / np.pi) + .1

    k_base = kerch.kernel.RBF(sample=sample)
    k = kerch.kernel.Nystrom(base_kernel=k_base)

    # kernel matrix
    fig1, axs1 = plt.subplots(2,2)
    fig1.suptitle('Kernel Matrices of the Base Kernel (RBF)')

    axs1[0,0].imshow(k_base.K, vmin=0, vmax=1)
    axs1[0,0].set_title("Sample - Sample")

    axs1[0,1].imshow(k_base.k(y=oos), vmin=0, vmax=1)
    axs1[0,1].set_title("Sample - OOS")

    axs1[1,0].imshow(k_base.k(x=oos), vmin=0, vmax=1)
    axs1[1,0].set_title("OOS - Sample")

    im1 = axs1[1,1].imshow(k_base.k(x=oos, y=oos), vmin=0, vmax=1)
    axs1[1,1].set_title("OOS - OOS")

    for ax in axs1.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.colorbar(im1, ax=axs1.ravel().tolist())

    # explicit feature map
    fig2, axs2 = plt.subplots(1,2)
    fig2.suptitle('Explicit Feature Maps (Nystrom)')

    axs2[0].imshow(k.Phi)
    axs2[0].set_title("Sample")

    im2 = axs2[1].imshow(k.phi(x=oos))
    axs2[1].set_title("OOS")

    for ax in axs2.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig2.colorbar(im2, ax=axs2.ravel().tolist())

    # kernel matrix from the explicit feature map
    fig3, axs3 = plt.subplots(2,2)
    fig3.suptitle('Kernel Matrices from the Explicit Feature Map (Nystrom)')

    axs3[0,0].imshow(k.K, vmin=0, vmax=1)
    axs3[0,0].set_title("Sample - Sample")

    axs3[0,1].imshow(k.k(y=oos), vmin=0, vmax=1)
    axs3[0,1].set_title("Sample - OOS")

    axs3[1,0].imshow(k.k(x=oos), vmin=0, vmax=1)
    axs3[1,0].set_title("OOS - Sample")

    im3 = axs3[1,1].imshow(k.k(x=oos, y=oos), vmin=0, vmax=1)
    axs3[1,1].set_title("OOS - OOS")

    for ax in axs3.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig3.colorbar(im3, ax=axs3.ravel().tolist())

Effect of the dimension
-----------------------
On the example hereabove, al lot of features seem to contain few information. One may decide that keeping all the bases of the RKHS is unnecessary, and only keeping the ones corresponding to the
most variance on the sample dataset. In the following example, only 6 dimensions are kept (few to accentuate the effect for demonstration purposes). This of course reduces the
quality of the approximation.

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    sample = np.sin(np.arange(0,15) / np.pi) + .1
    oos = np.sin(np.arange(15,30) / np.pi) + .1

    k_base = kerch.kernel.RBF(sample=sample)
    k = kerch.kernel.Nystrom(base_kernel=k_base, dim=6)

    # kernel matrix
    fig1, axs1 = plt.subplots(2,2)
    fig1.suptitle('Kernel Matrices of the Base Kernel (RBF)')

    axs1[0,0].imshow(k_base.K, vmin=0, vmax=1)
    axs1[0,0].set_title("Sample - Sample")

    axs1[0,1].imshow(k_base.k(y=oos), vmin=0, vmax=1)
    axs1[0,1].set_title("Sample - OOS")

    axs1[1,0].imshow(k_base.k(x=oos), vmin=0, vmax=1)
    axs1[1,0].set_title("OOS - Sample")

    im1 = axs1[1,1].imshow(k_base.k(x=oos, y=oos), vmin=0, vmax=1)
    axs1[1,1].set_title("OOS - OOS")

    for ax in axs1.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.colorbar(im1, ax=axs1.ravel().tolist())

    # explicit feature map
    fig2, axs2 = plt.subplots(1,2)
    fig2.suptitle('Explicit Feature Maps (Nystrom)')

    axs2[0].imshow(k.Phi)
    axs2[0].set_title("Sample")

    im2 = axs2[1].imshow(k.phi(x=oos))
    axs2[1].set_title("OOS")

    for ax in axs2.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig2.colorbar(im2, ax=axs2.ravel().tolist())

    # kernel matrix from the explicit feature map
    fig3, axs3 = plt.subplots(2,2)
    fig3.suptitle('Kernel Matrices from the Explicit Feature Map (Nystrom)')

    axs3[0,0].imshow(k.K, vmin=0, vmax=1)
    axs3[0,0].set_title("Sample - Sample")

    axs3[0,1].imshow(k.k(y=oos), vmin=0, vmax=1)
    axs3[0,1].set_title("Sample - OOS")

    axs3[1,0].imshow(k.k(x=oos), vmin=0, vmax=1)
    axs3[1,0].set_title("OOS - Sample")

    im3 = axs3[1,1].imshow(k.k(x=oos, y=oos), vmin=0, vmax=1)
    axs3[1,1].set_title("OOS - OOS")

    for ax in axs3.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig3.colorbar(im3, ax=axs3.ravel().tolist())


Using the factory
-----------------
The following codes are equivalent. First, an example with the default arguments:

.. code-block:: python

    k_base = kerch.kernel.RBF(sample=sample)
    k = kerch.kernel.Nystrom(base_kernel=k_base)


.. code-block:: python

    k_base = kerch.kernel.RBF(sample=sample)
    k = kerch.kernel.factory(kernel_type='nystrom', base_kernel=k_base)


.. code-block:: python

    k_base = kerch.kernel.factory(kernel_type='rbf')
    k = kerch.kernel.Nystrom(base_kernel=k_base)


.. code-block:: python

    k = kerch.kernel.factory(kernel_type='nystrom', base_type='rbf', sample=sample)


And now with some arguments:

.. code-block:: python

    k_base = kerch.kernel.RBF(sample=sample, sigma=2)
    k = kerch.kernel.Nystrom(base_kernel=k_base, dim=3)


.. code-block:: python

    k_base = kerch.kernel.RBF(sample=sample, sigma=2)
    k = kerch.kernel.factory(kernel_type='nystrom', base_kernel=k_base, dim=3)


.. code-block:: python

    k_base = kerch.kernel.factory(kernel_type='rbf', sample=sample, sigma=2)
    k = kerch.kernel.Nystrom(base_kernel=k_base, dim=3)


.. code-block:: python

    k = kerch.kernel.factory(kernel_type='nystrom', base_type='rbf', sample=sample, sigma=2, dim=3)


