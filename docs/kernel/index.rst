=============
Kernel Module
=============

This module contains many different types of kernels. Each kernel is created based on some hyperparameters and a sample
dataset.

If no sample dataset is provided, a random one will be initialized. This dataset can always be reinitialized
(`init_sample`) or the alue of the datapoints can be updated updated (`update_sample`). In the latter case, the
dimensions have to be matching. Furthermore, the sample dataset can also work in a stochastic manner, of which the
indices can be controlled through the `reset` method.

Both the value of the sample datapoints as the hyperparameters are compatible with gradient graphs of PyTorch. If such
a graph is to be computed, this has to be specifically specified during constructions.

All kernels can be centered, either implicitly using statistics on the kernel matrix of the sample dataset, either
explicitly using a statistic on the explicit feature map. In the former case, this cannot be extended to fully
out-of-sample computations.

At last, a Nystrom kernel is also implemented, which created an explicit feature map based on any kernel (possibly
implicit), using eigendocomposition. Among other things, this can serve as a solution for centering fully out-of-sample
kernel matrices of implicitly defined kernels.

The general structure of the module is based around an abstract kernel class `base`, of which
`rkm.kernle.implicit` and `explicit` inherit. All other kernels inherit of one of these two at the exception
of `polynomial` which directly inherits `base` as it has a primal formulation and a dual
formulation which can be computed otherwise than with an inner product of the explicit feature map.

Kernel Factory
--------------

.. autofunction:: rkm.kernel.factory

Generic Kernels
---------------

.. toctree::
    :maxdepth: 1

    linear
    rbf
    laplacian
    polynomial
    cosine
    sigmoid
    nystrom

Network-based kernels
---------------------

.. toctree::
    :maxdepth: 1

    explicit_nn
    implicit_nn

Time Kernels
------------

.. toctree::
    :maxdepth: 1

    indicator
    hat

Vision Kernels
--------------

.. toctree::
    :maxdepth: 1

    additive_chi2
    skewed_chi2

Abstract Kernels
----------------

.. toctree::
    :maxdepth: 1

    exponential
    explicit
    implicit
    base

