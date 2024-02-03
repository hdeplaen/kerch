=====================
Distance-Based Kernel
=====================

This class is meant to be inherited for to create kernels that are of the form

.. math::

    k(x,y) = f\left(\frac{d(x,y)}{\sigma}\right)

Abstract Classes
================

.. autoclass:: kerch.kernel.Distance
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
    :show-inheritance:

.. autoclass:: kerch.kernel.DistanceSquared
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
    :show-inheritance:



Defining a New Kernel
=====================

To create a new kernel, it suffices to provide an implementation of the private method ``_dist(self, x, y)``. The
automatic definition of the bandwidth and the other functionalities of all exponential kernel will be automatically
provided. The following examples show how to implement a kernel based on the :math:`\ell^1`-distance. In particular

.. math::
    k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_1}{2\sigma^2} \right).

Minimal Example
---------------

.. plot::
    :include-source:

    import kerch
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    # we define our l1 kernel
    class MyDistance(kerch.kernel.Distance):
        def _dist(self, x, y):
            # x: torch.Tensor of size [num_x, dim]
            # y: torch.Tensor of size [num_y, dim]
            x = x.T[:, :, None]
            y = y.T[:, None, :]

            diff = x - y

            # return torch.Tensor of size [num_x, num_y]
            return torch.sum(torch.abs(diff), dim=0, keepdim=False)

    # we define our sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # now we can just use the kernel
    k = MyExponential(sample=sample)

    plt.figure(1)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Sigma = "+str(k.sigma))

    k.sigma = 1

    plt.figure(2)
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Sigma = "+str(k.sigma))

This also works with the other properties of the kernels. In the following example, we have project all inputs linearly
in the interval :math:`[0,1]` (``minmax_rescaling``) and the feature map/kernel is centered.

.. code-block:: python

    # we define the sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # we also define an out_of_sample
    t_oos = np.expand_dims(np.arange(15,30), axis=1)
    oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

    # we initialize our new kernel
    k = MyExponential(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['center'])

    # sample
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(sample)
    axs[0].set_title("Original")
    im = axs[1].imshow(k.current_sample_projected)
    axs[1].set_title("Transformed")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    fig.suptitle('Sample')

    # out-of-sample
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(oos)
    axs[0].set_title("Original")
    im = axs[1].imshow(k.transform_input(oos))
    axs[1].set_title("Transformed")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    fig.suptitle('Out-of-Sample')

    # kernel matrix
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Kernel Matrix')

    axs[0,0].imshow(k.K, vmin=-1, vmax=1)
    axs[0,0].set_title("Sample - Sample")

    axs[0,1].imshow(k.k(y=oos), vmin=-1, vmax=1)
    axs[0,1].set_title("Sample - OOS")

    axs[1,0].imshow(k.k(x=oos), vmin=-1, vmax=1)
    axs[1,0].set_title("OOS - Sample")

    im = axs[1,1].imshow(k.k(x=oos, y=oos), vmin=-1, vmax=1)
    axs[1,1].set_title("OOS - OOS")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axs.ravel().tolist())


.. plot::

    import kerch
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    class MyExponential(kerch.kernel.Exponential):
        def _dist(self, x, y):
            # x: torch.Tensor of size [num_x, dim]
            # y: torch.Tensor of size [num_y, dim]
            x = x.T[:, :, None]
            y = y.T[:, None, :]

            diff = x - y

            # return torch.Tensor of size [num_x, num_y]
            return torch.sum(torch.abs(diff), dim=0, keepdim=False)

    # we define the sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # we also define an out_of_sample
    t_oos = np.expand_dims(np.arange(15,30), axis=1)
    oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

    # we initialize our new kernel
    k = MyExponential(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['center'])

    # sample
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(sample)
    axs[0].set_title("Original")
    im = axs[1].imshow(k.current_sample_projected)
    axs[1].set_title("Transformed")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    fig.suptitle('Sample')

    # out-of-sample
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(oos)
    axs[0].set_title("Original")
    im = axs[1].imshow(k.transform_input(oos))
    axs[1].set_title("Transformed")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    fig.suptitle('Out-of-Sample')

    # kernel matrix
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Kernel Matrix')

    axs[0,0].imshow(k.K, vmin=-1, vmax=1)
    axs[0,0].set_title("Sample - Sample")

    axs[0,1].imshow(k.k(y=oos), vmin=-1, vmax=1)
    axs[0,1].set_title("Sample - OOS")

    axs[1,0].imshow(k.k(x=oos), vmin=-1, vmax=1)
    axs[1,0].set_title("OOS - Sample")

    im = axs[1,1].imshow(k.k(x=oos, y=oos), vmin=-1, vmax=1)
    axs[1,1].set_title("OOS - OOS")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axs.ravel().tolist())


Extensive Example
-----------------
This is more extensive example where an extra parameter is added to control the degree of the distance. We also provide
a name and the ``hparams`` property.

.. code-block:: python

    import kerch
    import torch
    import numpy as np

    # we define our l1 kernel
    class MyExponential(kerch.kernel.Exponential):
        def __init__(self, *args, **kwargs):
            super(MyExponential, self).__init__(*args, **kwargs)

            # all parameters are typically passed through the keyword arguments kwargs
            degree = kwargs.pop('degree', 1)

            # we ensure that is a torch value of the correct data type used by kerch (modifiable)
            degree = kerch.utils.castf(degree, tensor=False)

            # we now store it as a parameter, tu ensure that the values are ported over when changing the device
            # ! due to the nature of PyTorch, parameters can only be added after the call to super.
            self.degree = torch.nn.Parameter(degree, requires_grad=False)

        def __str__(self):
            # it is always nicer to add a name to the kernel (it must begin with small letters, the capitalization is automatic)
            # this also has an influence on the __repr__ attribute
            return "l1 exponential kernel"

        @property
        def hparams(self) -> dict():
            # this returns a dictionary containing the properties
            # it is important to call the super to also pass the other parameters like the sigma etc.
            return {'Kernel': 'L1 Exponential',
                    'Degree': self.degree.detach().cpu().item().numpy()
                    **super(MyExponential, self).hparams}


        def _dist(self, x, y):
            # x: torch.Tensor of size [num_x, self.dim_input]
            # y: torch.Tensor of size [num_y, self.dim_input]
            x = x.T[:, :, None]
            y = y.T[:, None, :]

            diff = x - y

            # return torch.Tensor of size [num_x, num_y]
            return torch.sum(torch.abs(diff)**self.degree, dim=0, keepdim=False)



Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Distance
    kerch.kernel.DistanceSquared
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module