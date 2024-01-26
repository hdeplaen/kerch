===============
Implicit Kernel
===============

This class is meant to be inherited to create kernel that are defined implicitly with a :math:`k(x,y)`. These classes
do not have an explicit representation as it is supposed to live in a infinite-dimensional Hilbert space.

Abstract Class
==============

.. autoclass:: kerch.kernel.Implicit
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches, sample_as_param, phi_sample, phi, C
    :show-inheritance:

Defining a New Kernel
=====================

To create a new kernel, it suffices to provide an implementation of the private method ``_implicit(self, x, y)``.
By inheriting from ``kerch.kernel.Implicit``, all the functionalities of explicit kernels will be inherited. As an
example, let us implement the following kernel:

.. math::
    k(x,y) = \log(x^\top y + 1).

Minimal Example
---------------

.. plot::
    :include-source:

    import kerch
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    # we define our new kernel
    class MyImplicit(kerch.kernel.Implicit):
        def _implicit(self, x, y):
            # x: torch.Tensor of size [num_x, self.dim_input]
            # y: torch.Tensor of size [num_y, self.dim_input]

            k = torch.log(x @ y.T + 1)

            # return torch.Tensor of size [num_x, num_y]
            return k

    # now we can just use the kernel
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    k = MyImplicit(sample=sample)

    plt.figure()
    plt.imshow(k.K)
    plt.colorbar()
    plt.title("Kernel Matrix")


This also works with the other properties of the kernels. In the following example, we have project all inputs linearly
in the interval :math:`[0,1]` (``minmax_rescaling``) and the feature map/kernel is centered and then normalized.

.. code-block:: python

    # we define the sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # we also define an out_of_sample
    t_oos = np.expand_dims(np.arange(15,30), axis=1)
    oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

    # we initialize our new kernel
    k = MyImplicit(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['center','normalize'])

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

    # we define our new kernel
    class MyImplicit(kerch.kernel.Implicit):
        def _implicit(self, x, y):
            # x: torch.Tensor of size [num_x, self.dim_input]
            # y: torch.Tensor of size [num_y, self.dim_input]

            k = torch.log(x @ y.T + 1)

            # return torch.Tensor of size [num_x, num_y]
            return k

    # we define the sample
    t = np.expand_dims(np.arange(0,15), axis=1)
    sample = np.concatenate((np.sin(t / np.pi) + 1, np.cos(t / np.pi) - 1), axis=1)

    # we also define an out_of_sample
    t_oos = np.expand_dims(np.arange(15,30), axis=1)
    oos = np.concatenate((np.sin(t_oos / np.pi) + 1, np.cos(t_oos / np.pi) - 1), axis=1)

    # we initialize our new kernel
    k = MyImplicit(sample=sample, sample_transform=['minmax_rescaling'], kernel_transform=['center','normalize'])

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
This is more extensive example where an extra parameter :math:`a` is added. We also provide
a name and the ``hparams`` property. The extended kernel definition is

.. math::

    k(x,y) = \log(x^\top y + a).


Because of the logarithm, we must also verify that :math:`a` is strictly positive (inner product are guaranteed positive).

.. code-block:: python

    import kerch
    import torch
    import numpy as np

    # we define our new implicit kernel
    class MyImplicit(kerch.kernel.Exponential):
        def __init__(self, *args, **kwargs):
            # all parameters are typically passed through the keyword arguments kwargs
            a = kwargs.pop('a', 1)

            # we assert that the value is positive because of the logarithm.
            assert a > 0, 'The value for the parameter a must be strictly positive.'

            # we ensure that is a torch value of the correct data type used by kerch (modifiable)
            a = kerch.utils.castf(a, tensor=False)

            # call to super
            super(MyImplicit, self).__init__(*args, **kwargs)

            # we now store it as a parameter, tu ensure that the values are ported over when changing the device
            # ! due to the nature of PyTorch, parameters can only be added after the call to super.
            self.a = torch.nn.Parameter(a, requires_grad=False)

        def __str__(self):
            # it is always nicer to add a name to the kernel (it must begin with small letters, the capitalization is automatic)
            # this also has an influence on the __repr__ attribute
            return "my implicit kernel"

        @property
        def hparams(self) -> dict():
            # this returns a dictionary containing the properties
            # it is important to call the super to also pass the other parameters like the sigma etc.
            return {'Kernel': 'MyImplicit',
                    'Kernel parameter a': self.a.detach().cpu().item().numpy()
                    **super(MyExplicit, self).hparams}


        def _implicit(self, x, y):
            # x: torch.Tensor of size [num_x, self.dim_input]
            # y: torch.Tensor of size [num_y, self.dim_input]

            k = torch.log(x @ y.T + self.a)

            # return torch.Tensor of size [num_x, num_y]
            return k



Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Implicit
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module