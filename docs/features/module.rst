============
Kerch Module
============

The :py:class:`kerch.feature.Module` class is aimed at modules that must be trained trough gradient descent. It extends the
:external+torch:py:class:`torch.nn.modules.module.Module` to add the logging
features of the :py:class:`kerch.feature.Logger` class.

Functionalities
===============

It also adds the following functionalities necessary for more complex gradient descent than what PyTorch offers. In
particular:

Before and After Step Operations
--------------------------------

The methods :py:meth:`~kerch.feature.Module.before_step` and :py:meth:`~kerch.feature.Module.after_step` for
operations that must be executed before and after a parameter update through an optimization step.

Different Parameter Types
-------------------------
The support for various groups of parameters that require specific learning rates or lie on the Stiefel manifold. The
following group types are available

* Euclidean
    Parameters lying on the Euclidean manifold (standard optimization). The optimization is
    done onto :math:`\mathbb{R}^{n \times m}`, :math:`n` and :math:`m` depending on the size of each parameter.
* Stiefel
    Parameters that must lie on the Stiefel manifold (optimization is done onto that manifold).
    The Stiefel manifold corresponds to the orthonormal parameters :math:`U \in \mathrm{St}(n,m)`, i.e., all
    :math:`U \in \mathbb{R}^{n \times m}` such that :math:`U^\top U = I`. The dimensions :math:`n` and :math:`m` are
    proper to each parameter.
* Slow
    Parameters lying on the Euclidean manifold (standard optimization). The optimization is
    done onto :math:`\mathbb{R}^{n \times m}`, :math:`n` and :math:`m` depending on the size of each parameter.
    The specificity of these slow Euclidean parameters is that they are better trained with a lower learning rate that the
    others, hence their name and the necessity to group them apart.

Hyperparameters Dictionaries
----------------------------
This is relevant for automatically recording values before, during of after the training. All the relevant hyperparameters are
listed into two dictionaries.

* Fixed Hyperparameters
    The attribute :py:attr:`~kerch.feature.Module.hparams_fixed` return the fixed hyperparameters of the module. By contrast with :py:attr:`hparams_variable`, these are the values that are fixed and
    cannot possibly change during the training. If applicable, these can be specific architecture values for example.
* Variable Hyperparameters
    The attribute :py:attr:`~kerch.feature.Module.hparams_variable` return the fixed hyperparameters of the module. By contrast with :py:attr:`hparams_fixed`, these are the values that are may change during
    the training and may be monitored at various stages during the training. If applicable, these can be kernel bandwidth parameters for example.


Abstract Class
==============

.. autoclass:: kerch.feature.Module
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches
    :private-members: _euclidean_parameters, _stiefel_parameters, _slow_parameters
    :show-inheritance:


Examples
========

KPCA
----

In the following example, we create a :py:class:`kerch.level.KPCA` level based on a :py:class:`kerch.kernel.RBF` kernel
where we specify that only the level parameters are trainable by gradient descent.

.. code-block::

    import kerch
    import torch

    x = torch.randn(5, 3)
    kpca = kerch.level.KPCA(sample=x,                 # random sample
                            kernel_type='rbf',        # we use a rbf kernel (this is the default value, but we specify it for clarity)
                            sigma=2,                  # we specify a RBF bandwidth value
                            representation='dual',    # we work in dual representation (also default value, but specified for clarity)
                            dim_output=2,             # we want an output dimension of 2 (the input is 3)
                            sample_trainable=False,   # the sample can be trained, but we don't want that: we want it fixed
                            sigma_trainable=False,    # the sigma can also be trained, but we don't want that either
                            level_trainable=True)     # the level parameters are trainable, meaning that the eigenvectors are trainable by gradient

We indeed see that from all
parameters printed, only the eigenvectors :py:attr:`~kerch.level.KPCA.hidden` (lying on the Stiefel manifold) have ``requires_grad=True``. The Euclidean
parameters correspond here to the sample :py:attr:`~kerch.feature.Sample.sample`, that we do not want to be trained. The slow (Euclidean) parameters correspond
to the bandwidth of the kernel :py:attr:`~kerch.kernel.RBF.sigma`, which we also want to remain fixed and not optimized through gradient descent.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    x = torch.randn(5, 3)
    kpca = kerch.level.KPCA(sample=x,                 # random sample
                            kernel_type='rbf',        # we use a rbf kernel (this is the default value, but we specify it for clarity)
                            sigma=2,                  # we specify a RBF bandwidth value
                            representation='dual',    # we work in dual representation (also default value, but specified for clarity)
                            dim_output=2,             # we want an output dimension of 2 (the input is 3)
                            sample_trainable=False,   # the sample can be trained, but we don't want that: we want it fixed
                            sigma_trainable=False,    # the sigma can also be trained, but we don't want that either
                            level_trainable=True)     # the level parameters are trainable, meaning that the eigenvectors are trainable by gradient
    # --- hide: stop ---

    # Euclidean parameters
    print('EUCLIDEAN PARAMETERS:')
    for p in kpca.manifold_parameters(type='euclidean'):
        print(p)

    # Stiefel parameters
    print('\nSTIEFEL PARAMETERS:')
    for p in kpca.manifold_parameters(type='stiefel'):
        print(p)

    # Slow (Euclidean) parameters
    print('\nSLOW (EUCLIDEAN) PARAMETERS:')
    for p in kpca.manifold_parameters(type='slow'):
        print(p)

We can have a look at the hyperparameters. The parameter ``sigma``, even if not trainable is always listed in the
variable hyperparameters. Its value will just not change during the training.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    x = torch.randn(5, 3)
    kpca = kerch.level.KPCA(sample=x,                 # random sample
                            kernel_type='rbf',        # we use a rbf kernel (this is the default value, but we specify it for clarity)
                            sigma=2,                  # we specify a RBF bandwidth value
                            representation='dual',    # we work in dual representation (also default value, but specified for clarity)
                            dim_output=2,             # we want an output dimension of 2 (the input is 3)
                            sample_trainable=False,   # the sample can be trained, but we don't want that: we want it fixed
                            sigma_trainable=False,    # the sigma can also be trained, but we don't want that either
                            level_trainable=True)     # the level parameters are trainable, meaning that the eigenvectors are trainable by gradient
    # --- hide: stop ---

    print('FIXED HYPERPARAMETERS:')
    for key, value in kpca.hparams_fixed.items():
        print(key, ":", value)

    print('\nVARIABLE HYPERPARAMETERS:')
    for key, value in kpca.hparams_variable.items():
        print(key, ":", value)



Creating a Module
-----------------

In this example, we create a module containing a parameter on the Euclidean manifold. We therefore overwrite the
:py:meth:`~kerch.feature.Module._euclidean_parameters` method and not forget to call the inherited classes to not forget to return all
parameters returned by the mother classes.

.. code-block::

    import kerch
    import torch
    from typing import Iterator


    class MyModule(kerch.feature.Module):
        def __init__(self, *args, **kwargs):
            super(MyModule, self).__init__(*args, **kwargs)

            # we recover the parameter size by the argument param_size
            param_size = kwargs.pop('param_size', (1, 1))

            # we create our parameter of float type kerch.FTYPE
            # (this value can be modified and ensures that all floating types are the same throughout the code)
            self.my_param = torch.nn.Parameter(torch.randn(param_size, dtype=kerch.FTYPE), requires_grad=True)

        def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
            # important not to forget, otherwise the parameters returned by mother classes will be skipped
            yield from super(MyModule, self)._euclidean_parameters(recurse=recurse)

            # we yield our additional new parameter
            yield self.my_param

        def after_step(self):
            # after each training step, we want the columns to be centered
            with torch.no_grad():
                self.my_param.data = self.my_param - torch.mean(self.my_param, dim=0)

        @property
        def hparams_fixed(self) -> dict:
            # we add the shape of our parameter to the fixed hyperparameters
            # we don't forget to return the other possible hyperparameters issued by parent classes
            return {'my_param size': self.my_param.shape,
                    **super(MyModule, self).hparams_fixed}

    # We instantiate our class
    my_module = MyModule(param_size=(2, 3))


We can have a look at the parameters. The same can be done for the :py:meth:`~kerch.feature.Module._stiefel_parameters` and
:py:meth:`~kerch.feature.Module._slow_parameters` methods.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch
    from typing import Iterator

    torch.manual_seed(0)

    class MyModule(kerch.feature.Module):
        def __init__(self, *args, **kwargs):
            super(MyModule, self).__init__(*args, **kwargs)

            # we recover the parameter size by the argument param_size
            param_size = kwargs.pop('param_size', (1, 1))

            # we create our parameter of float type kerch.FTYPE
            # (this value can be modified and ensures that all floating types are the same throughout the code)
            self.my_param = torch.nn.Parameter(torch.randn(param_size, dtype=kerch.FTYPE), requires_grad=True)

        def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
            # important not to forget, otherwise the parameters returned by mother classes will be skipped
            yield from super(MyModule, self)._euclidean_parameters(recurse=recurse)

            # we yield our additional new parameter
            yield self.my_param

        def after_step(self):
            # after each training step, we want the columns to be centered
            with torch.no_grad():
                self.my_param.data = self.my_param - torch.mean(self.my_param, dim=0)

        @property
        def hparams_fixed(self) -> dict:
            # we add the shape of our parameter to the fixed hyperparameters
            # we don't forget to return the other possible hyperparameters issued by parent classes
            return {'my_param size': self.my_param.shape,
                    **super(MyModule, self).hparams_fixed}

    # We instantiate our class
    my_module = MyModule(param_size=(2, 3))
    # --- hide: stop ---

    # Euclidean parameters
    print('EUCLIDEAN PARAMETERS:')
    for p in my_module.manifold_parameters(type='euclidean'):
        print(p)

    # Stiefel parameters
    print('\nSTIEFEL PARAMETERS:')
    for p in my_module.manifold_parameters(type='stiefel'):
        print(p)

    # Slow (Euclidean) parameters
    print('\nSLOW (EUCLIDEAN) PARAMETERS:')
    for p in my_module.manifold_parameters(type='slow'):
        print(p)


If :py:meth:`~kerch.feature.Module.after_step` is called, we can observe that the parameter is centered along the columns.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch
    from typing import Iterator

    torch.manual_seed(0)

    class MyModule(kerch.feature.Module):
        def __init__(self, *args, **kwargs):
            super(MyModule, self).__init__(*args, **kwargs)

            # we recover the parameter size by the argument param_size
            param_size = kwargs.pop('param_size', (1, 1))

            # we create our parameter of float type kerch.FTYPE
            # (this value can be modified and ensures that all floating types are the same throughout the code)
            self.my_param = torch.nn.Parameter(torch.randn(param_size, dtype=kerch.FTYPE), requires_grad=True)

        def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
            # important not to forget, otherwise the parameters returned by mother classes will be skipped
            yield from super(MyModule, self)._euclidean_parameters(recurse=recurse)

            # we yield our additional new parameter
            yield self.my_param

        def after_step(self):
            # after each training step, we want the columns to be centered
            with torch.no_grad():
                self.my_param.data = self.my_param - torch.mean(self.my_param, dim=0)

        @property
        def hparams_fixed(self) -> dict:
            # we add the shape of our parameter to the fixed hyperparameters
            # we don't forget to return the other possible hyperparameters issued by parent classes
            return {'my_param size': self.my_param.shape,
                    **super(MyModule, self).hparams_fixed}

    my_module = MyModule(param_size=(2, 3))

    # --- hide: stop ---

    # we suppose that an optimization step has been performed
    my_module.after_step()

    # Euclidean parameters
    print('EUCLIDEAN PARAMETERS:')
    for p in my_module.manifold_parameters(type='euclidean'):
        print(p)

Similarly, let us print the hyperparameters:

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch
    from typing import Iterator

    torch.manual_seed(0)

    class MyModule(kerch.feature.Module):
        def __init__(self, *args, **kwargs):
            super(MyModule, self).__init__(*args, **kwargs)

            # we recover the parameter size by the argument param_size
            param_size = kwargs.pop('param_size', (1, 1))

            # we create our parameter of float type kerch.FTYPE
            # (this value can be modified and ensures that all floating types are the same throughout the code)
            self.my_param = torch.nn.Parameter(torch.randn(param_size, dtype=kerch.FTYPE), requires_grad=True)

        def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
            # important not to forget, otherwise the parameters returned by mother classes will be skipped
            yield from super(MyModule, self)._euclidean_parameters(recurse=recurse)

            # we yield our additional new parameter
            yield self.my_param

        def after_step(self):
            # after each training step, we want the columns to be centered
            with torch.no_grad():
                self.my_param.data = self.my_param - torch.mean(self.my_param, dim=0)

        @property
        def hparams_fixed(self) -> dict:
            # we add the shape of our parameter to the fixed hyperparameters
            # we don't forget to return the other possible hyperparameters issued by parent classes
            return {'my_param size': self.my_param.shape,
                    **super(MyModule, self).hparams_fixed}

    my_module = MyModule(param_size=(2, 3))

    # --- hide: stop ---

    print('FIXED HYPERPARAMETERS:')
    for key, value in my_module.hparams_fixed.items():
        print(key, ":", value)

    print('\nVARIABLE HYPERPARAMETERS:')
    for key, value in my_module.hparams_variable.items():
        print(key, ":", value)

Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.feature.Module
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module