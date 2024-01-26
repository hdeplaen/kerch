============
Kernel Class
============

This class is meant to create kernels that have an explicit feature map :math:`\phi(x)`, but also another way of computing the kernel
:math:`k(x,y)` than through the inner product of the feature maps. This is for example the case of the :class:`kerch.kernel.Polynomial` kernel.
When inheriting from this class, both methods ``_explicit(self, x)``, ``_implicit(self, x, y)`` and the property ``_dim_feature``
need to be defined. Beware however of inconsistencies as depending on the representation, the kernel will be computed either
implicitly with the provided method or explicitly as the inner product of the explicit feature map defined by the other method.
If you only want to define one of both method, it is preferable directly use :class:`kerch.kernel.Implicit` or :class:`kerch.kernel.Explicit`
to avoid possible inconsistencies. We also refer to these classes for the correct implementation of the necessary methods.

Abstract Class
--------------

.. autoclass:: kerch.kernel.Kernel
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches
    :show-inheritance:


Mother Class
------------
This abstract class is itself built onto an even more abstract class that does not support kernel transforms. One can
inherit from that one if no transforms are required. The cache management is also minimal.

.. autoclass:: kerch.kernel._BaseKernel
    :members:
    :inherited-members: Module
    :undoc-members:
    :exclude-members: training, dump_patches
    :show-inheritance:





Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.kernel.Kernel
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module
