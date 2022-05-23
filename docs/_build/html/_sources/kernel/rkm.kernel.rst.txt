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

The general structure of the module is based around an abstract kernel class `rkm.kernel.base`, of which
`rkm.kernle.implicit` and `rkm.kernel.explicit` inherit. All other kernels inherit of one of these two at the exception
of `rkm.kernel.polynomial` which directly inherits `rkm.kernel.base` as it has a primal formulation and a dual
formulation which can be computed otherwise than with an inner product of the explicit feature map.

Kernel Factory
--------------

.. autofunction:: rkm.kernel.factory

Generic Kernels
---------------

.. toctree::
    rkm.kernel.linear
    rkm.kernel.rbf
    rkm.kernel.polynomial
    rkm.kernel.cosine
    rkm.kernel.sigmoid
    rkm.kernel.nystrom

Network-based kernels
---------------------

.. toctree::
    rkm.kernel.explicit_nn
    rkm.kernel.implicit_nn

Time Kernels
------------

.. toctree::
    rkm.kernel.indicator
    rkm.kernel.hat

Vision Kernels
--------------

.. toctree::
    rkm.kernel.additive_chi2
    rkm.kernel.skewed_chi2


Abstract Kernels
----------------

.. toctree::
    rkm.kernel.explicit
    rkm.kernel.implicit
    rkm.kernel.base

