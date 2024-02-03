===============
Explicit Kernel
===============

This class is meant to be inherited to create kernels that are defined by an explicit feature map :math:`\phi(x)`. The
kernel will be automatically computed as the inner product of the feature maps :math:`k(x,y)=\phi(x)^\top\phi(y)`
without it having to be defined in the inherited implementation.

Abstract Class
==============

.. autoclass:: kerch.kernel.Explicit
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches
    :show-inheritance:

Defining a New Kernel
=====================

To create a new kernel, it suffices to provide an implementation of the private method ``_explicit(self, x)``.
By inheriting from ``kerch.kernel.Explicit``, all the functionalities of explicit kernels will be inherited. As an
example, let us implement the following kernel:

.. math::
    \phi(x) = \left[x_1, \ldots, x_{\texttt{num}}, x_1^2, \ldots, x_{\texttt{num}}^2, \log(x^\top x + 1)\right].

Minimal Example
---------------

.. plot::
    :include-source:

    import kerch
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    # we define our new kernel
    class MyExplicit(kerch.kernel.Explicit):
        def _explicit(self, x):
            # x: torch.Tensor of size [num, self.dim_input]
            phi1 = x                                                    # [num, self.dim_input]
            phi2 = x ** 2                                               # [num, self.dim_input]
            phi3 = torch.log(torch.sum(x * x, dim=1, keepdim=True)+1)   # [num, 1]

            phi = torch.cat((phi1, phi2, phi3), dim=1)                  # [num, 2*self.dim_input + 1]

            # return torch.Tensor of size [num, self.dim_feature]
            # if not specified (see further), self.dim_feature will be determined automatically
            return phi

    # now we can just use the kernel
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    k = MyExplicit(sample=sample)


    fig, axs = plt.subplots(1,2)
    axs[0].imshow(sample)
    axs[0].set_title("Sample")
    im = axs[1].imshow(k.Phi)
    axs[1].set_title("Feature Map")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')


    plt.figure()
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Kernel Matrix")


This also works with the other properties of the kernels. In the following example, we have project all inputs linearly
in the interval :math:`[0,1]` (``'minmax_rescaling'``) and the feature map is standardized (``'standard'``).

.. code-block:: python

    # we define the sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # we also define an out_of_sample
    t_oos = np.expand_dims(np.arange(15,30), axis=1)
    oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

    # we initialize our new kernel
    k = MyExplicit(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['standard'])

    # sample
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(sample)
    axs[0].set_title("Original")
    axs[1].imshow(k.current_sample_projected)
    axs[1].set_title("Transformed")
    im = axs[2].imshow(k.Phi)
    axs[2].set_title("Feature Map")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    fig.suptitle('Sample')

    # out-of-sample
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(oos)
    axs[0].set_title("Original")
    axs[1].imshow(k.transform_input(oos))
    axs[1].set_title("Transformed")
    im = axs[2].imshow(k.phi(x=oos))
    axs[2].set_title("Feature Map")
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

    # we define our new kernel
    class MyExplicit(kerch.kernel.Explicit):
        def _explicit(self, x):
            # x: torch.Tensor of size [num, dim]
            phi1 = x                                                    # [num, dim]
            phi2 = x ** 2                                               # [num, dim]
            phi3 = torch.log(torch.sum(x * x, dim=1, keepdim=True)+1)   # [num, 1]

            phi = torch.cat((phi1, phi2, phi3), dim=1)                  # [num, 2*dim + 1]

            # return torch.Tensor of size [num, dim_feature]
            # if not specified (see further), dim_feature will be determined automatically
            return phi

    # we define the sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # we also define an out_of_sample
    t_oos = np.expand_dims(np.arange(15,30), axis=1)
    oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

    # we initialize our new kernel
    k = MyExplicit(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['standard'])

    # sample
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(sample)
    axs[0].set_title("Original")
    axs[1].imshow(k.current_sample_projected)
    axs[1].set_title("Transformed")
    im = axs[2].imshow(k.Phi)
    axs[2].set_title("Feature Map")
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    fig.suptitle('Sample')

    # out-of-sample
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(oos)
    axs[0].set_title("Original")
    axs[1].imshow(k.transform_input(oos))
    axs[1].set_title("Transformed")
    im = axs[2].imshow(k.phi(x=oos))
    axs[2].set_title("Feature Map")
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
This is more extensive example where an extra parameter is added to control the degree of the exponentiation. We also provide
a name and the ``hparams`` property. We also define ``dim_feature``.

.. code-block:: python

    import kerch
    import torch
    import numpy as np

    # we define our new kernel
    class MyExplicit(kerch.kernel.Exponential):
        def __init__(self, *args, **kwargs):
            # all parameters are typically passed through the keyword arguments kwargs
            degree = kwargs.pop('degree', 1)

            # call to super
            super(MyExplicit, self).__init__(*args, **kwargs)

            # we ensure that is a torch value of the correct data type used by kerch (modifiable)
            degree = kerch.utils.castf(degree, tensor=False)

            # we now store it as a parameter, tu ensure that the values are ported over when changing the device
            # ! due to the nature of PyTorch, parameters can only be added after the call to super.
            self.degree = torch.nn.Parameter(degree, requires_grad=False)

        def __str__(self):
            # it is always nicer to add a name to the kernel (it must begin with small letters, the capitalization is automatic)
            # this also has an influence on the __repr__ attribute
            return "my explicit kernel"

        @property
        def dim_feature(self) -> int:
            # it is more efficient to provide the feature dimension explicitly to avoid the model determining it on the
            # running the explicit method on the sample, which is a rather useless computation
            return 2 * self.dim_input + 1

        @property
        def hparams(self) -> dict():
            # this returns a dictionary containing the properties
            # it is important to call the super to also pass the other parameters like the sigma etc.
            return {'Kernel': 'MyExplicit',
                    'Degree': self.degree.detach().cpu().item().numpy()
                    **super(MyExplicit, self).hparams}


        def _explicit(self, x):
            # x: torch.Tensor of size [num, self.dim_input]
            phi1 = x                                                    # [num, self.dim_input]
            phi2 = x ** self.degree                                     # [num, self.dim_input]
            phi3 = torch.log(torch.sum(x * x, dim=1, keepdim=True)+1)   # [num, 1]

            phi = torch.cat((phi1, phi2, phi3), dim=1)                  # [num, 2 * self.dim_input + 1]

            # return torch.Tensor of size [num, self.dim_feature]
            # self.dim_feature is provided so it will be deduced from the provided definition and not computed on the go
            return phi



Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Explicit
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module