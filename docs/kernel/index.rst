=============
Kernel Module
=============

.. py::module:: kernel

Introduction
============

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
`kerch.kernle.implicit` and `explicit` inherit. All other kernels inherit of one of these two at the exception
of `polynomial` which directly inherits `base` as it has a primal formulation and a dual
formulation which can be computed otherwise than with an inner product of the explicit feature map.


Kernel Factory
--------------

.. autofunction:: kerch.kernel.factory

Examples
========

.. plot::
    :include-source:

    import kerch
    import numpy as np
    from matplotlib import pyplot as plt

    sample = np.sin(np.arange(0,15) / np.pi) + .1
    oos = np.sin(np.arange(15,30) / np.pi) + .1

    k = kerch.kernel.factory(type="polynomial", sample=sample, center=True, normalize=True)

    fig, axs = plt.subplots(2,2)

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



Different Kernels
=================

Generic Kernels
---------------

.. toctree::
    :maxdepth: 3

    linear
    rbf
    laplacian
    polynomial
    cosine
    sigmoid
    rff
    nystrom

Network-based kernels
---------------------

.. toctree::
    :maxdepth: 3

    explicit_nn
    implicit_nn

Time Kernels
------------

The idea behind time kernels is that time has the same local effect at
all time, or in other words that the kernels are translational invariant. We typically consider the following kernels:

.. toctree::
    :maxdepth: 3

    indicator
    hat
    rbf


Vision Kernels
--------------

.. toctree::
    :maxdepth: 3

    additive_chi2
    skewed_chi2

Abstract Kernels
----------------

.. toctree::
    :maxdepth: 3

    exponential
    explicit
    implicit
    kernel


Creating Kernels
================

It is also possible to create other kernels than the ones already implemented and still taking benefit of all features
common to all kernel such as automatic transformation of the input of the kernel (centering, normalization...), all
attributes (``K``, ``Phi``...), cache management, stochasicity etc. We refer to the examples in the documentation of the abstract
classes :class:`kerch.kernel.Exponential`, :class:`kerch.kernel.Explicit`, :class:`kerch.kernel.Implicit` and
:class:`kerch.kernel.Kernel`.

..
    Inheritance Diagram
    ====================
    .. inheritance-diagram::
        kerch.kernel.RBF
        kerch.kernel.Polynomial
        kerch.kernel.Indicator
        kerch.kernel.Cosine
        kerch.kernel.Linear
        kerch.kernel.Nystrom
        kerch.kernel.AdditiveChi2
        kerch.kernel.Explicit
        kerch.kernel.ExplicitNN
        kerch.kernel.Exponential
        kerch.kernel.Hat
        kerch.kernel.ImplicitNN
        kerch.kernel.RFF
        kerch.kernel.Sigmoid
        kerch.kernel.SkewedChi2
        kerch.kernel.Laplacian
        :private-bases:
        :top-classes: kerch.feature.Logger torch.nn.Module
