================
Cache Management
================

When performing various operations on the Kerch modules, it may happen that some results are necessary multiple times.
Some of these operations are however expensive to compute and it would be ideal to avoid recomputation and load them
from memory when already computed previously. This justifies the addition of a cache manager. The purpose of the
:py:class:`kerch.feature.Cache` class is to extend the :py:class:`kerch.feature.Module` class with a cache manager.
To illustrate its relevance, we can consider two example use cases:

* Kernel Matrix:
    Due to its quadratic complexity, computing the kernel matrix may be very expensive, in particular if
    not computed through explicit feature maps. Let us suppose that the kernel matrix has already been computed
    in order to be plotted for example. If one wants to then compute its eigendecomposition for KPCA, the matrix is
    reloaded from memory and not computed a second time.

* Data Transformations:
    We consider a big sample dataset that we want to be centered and normalized. When working with out-of-sample
    datasets, these have to be centered using the same statistics as sample in order to keep the model consistent.
    These statistics can be stored and re-used every time computations have to be performed on out-of-sample datasets.

In other words, this is a way to bypass the garbage-collector, but with a lot of granularity.

Functionalities
===============

Automatic Getter and Saver
--------------------------
Essentially most of the work is done through the :py:meth:`~kerch.feature.Cache._get` method that allows to check in
one line if the result of an operation (a kernel matrix, sample statistics...) have already been computed and return
it. If not computed, the information is stored in the cache, ready for further re-use (through the
:py:meth:`~kerch.feature.Cache._save` method that is automatically called by :py:meth:`~kerch.feature.Cache._get` in
if not already present in the cache).

Cache Levels
------------
It would be totally inefficient to store everything in the cache. Therefore, different cache levels exist meant for
storing different values. Each value is assigned to a specific value when computed for the first time. Each module
has a internal cache level (:py:attr:`~kerch.feature.Cache.cache_level`) that serves as a threshold when new values
are computed. If the specified cache level of the newly computed result exceeds the default cache level of the module,
the information is not saved. The next time that the same computation is required, it will thus not be loaded from the
cache, but computed again. We distinguish the following cache levels:

* ``"none"``: the cache is non-existent and everything is computed on the go. This is the lightest for the memory.
* ``"light"``: the cache is very light. For example, only the kernel matrix and statistics of the sample points are saved.
* ``"normal"``: same as light, but the statistics of the out-of-sample points are also saved, not the kernel matrices.
* ``"heavy"``: in addition to the statistics, the final kernel matrices of the out-of-sample points are saved.
* ``"total"``: every step of any computation is saved. This is very heavy on the memory.

The higher the cache level, the more will be stored into memory and the less redundancy will be introduced in the need
for recomputations. The module's :py:attr:`~kerch.feature.Cache.cache_level` attribute therefore controls a time versus
memory trade-off.

Resetter
--------
If the sample changes for example, most of the cache entries require te be recomputed. The two
private methods :py:meth:`~kerch.feature.Cache._reset_cache` and :py:meth:`~kerch.feature.Cache._clean_cache` therefore
exists to reset and clean the cache. This is done automatically when necessary. In practice, unless wanting to tweak
the package's internal working, these methods should not be called by the user. If the user really wants to reset the
cache, he may use the :py:meth:`~kerch.feature.Cache.reset` method. He may also visualize the current cache entries by
calling the :py:meth:`~kerch.feature.Cache.print_cache` method or the :py:meth:`~kerch.feature.Cache.cache_keys`
method to retrieve the keys of the cached values.

Default Cache Levels
====================

For each possible value to be stored, default cache levels are saved defined in :py:data:`kerch.DEFAULT_CACHE_LEVELS`.
These values can be changed if the user wants to customize the granularity. Here follows a summary.

Sample-Specific
---------------
The Transformation Tree refers to an instance of :py:class:`kerch.transform.TransformTree` and does not hold any data in
itself apart from the successive operations in themselves to perform the transformations, i.e., the tree of the transformations.
We refer further the cache levels inside the tree.

.. include:: cache_levels/sample.rst

Kernel-Specific
---------------

This is relevant for all classes who inherit from :py:class:`kerch.kernel.Kernel`. The attributes :py:attr:`~kerch.kernel.Kernel.K`,
:py:attr:`~kerch.kernel.Kernel.Phi` and :py:attr:`~kerch.kernel.Kernel.C` are always saved once computed until the sample
changes. The Transformation Trees refer to instances :py:class:`kerch.transform.TransformTree` (one for the explicit and one for the implicit)
and do not hold any data in
themselves apart from the successive operations themselves to perform the transformations, i.e., the tree of the transformations.
We refer further the cache levels inside the trees.

.. include:: cache_levels/kernel.rst


Transformation-Specific
-----------------------

The tree itself contains the transformations in themselves. These values refer to which values are stored inside a
transformation tree instance :py:class:`kerch.transform.TransformTree`. The tree can store both the statistics
(average, variance, minimum...) required to perform the transformations and the transformed values themselves
(centered valued, normalized values...). De `default` refers to the default transformation. We refer to the
documentation of :doc:`../features/transform` for further information.

.. include:: cache_levels/transform.rst

Level-Specific
--------------

The output of a level is also saved by default, until the sample or the model parameters change. The default representation
refers to ``primal`` or ``dual``. We refer to the documentation of the :doc:`../level/index` for further information.
Many models require an identity matrix to solve the model. This matrix can be stored for further usage, unless the
dimensions change. The different constituents of the loss (regularization term, recontruction term...), referred to as `sublosses`, are saved
independently from the total loss for monitoring. These are also resetted once the :py:meth:`~kerch.feature.Module.before_step`
method is called.

.. include:: cache_levels/level.rst


Abstract Class
==============


.. autoclass:: kerch.feature.Cache
    :members:
    :inherited-members: Module
    :undoc-members:
    :private-members: _get, _save, _clean_cache, _reset_cache, _remove_from_cache, _apply
    :exclude-members: training, dump_patches
    :show-inheritance:


Examples
========

For a proper usage of the Kerch package, there is no need to manage the cache. For the sake of completeness, we however
provide two examples. The first one shows how the cache works on an existing implementation. The second example shows
how one can manage cache elements by itself.

KPCA
----

The following example illustrates the working of the cache. We will consider two examples of a
:py:class:`kerch.level.KPCA`, one with a ``light`` cache level and another with a ``total`` cache level.

.. code-block::

    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_light_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='light')            # a 'light' cache level (only related to the sample)

    kpca_total_cache = kerch.level.KPCA(sample=sample,                  # idem
                                        dim_output=2,                   # idem
                                        sample_transform=['min'],       # idem
                                        kernel_transform=['center'],    # idem
                                        cache_level='total')            # a 'total' cache level (saves everything)


If we plot the cache now, nothing is printed: the cache is empty. The advantage of this package is that it does not perform any
unnecessary computations. Depending of what is required, it will compute what is strictly necessary.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_light_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='light')            # a 'light' cache level (only related to the sample)

    # --- hide: stop ---

    kpca_light_cache.print_cache()


After solving the model and passing an out-of-sample through the model, we can see that the cache is pretty much
loaded. Nothing however has been saved on related to the out-of-sample, even if it has been computed. This is a
consequence of ``light`` cache level of the module. Similarly, nor has the original non-centered kernel matrix, nor has
the original non-transformed sample been saved.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_light_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='light')            # a 'light' cache level (only related to the sample)

    # --- hide: stop ---

    kpca_light_cache.solve()
    kpca_light_cache.forward(oos)
    kpca_light_cache.print_cache()


We can optionally reset the cache. Nothing is printed: the cache is empty again.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_light_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='light')            # a 'light' cache level (only related to the sample)

    kpca_light_cache.solve()
    kpca_light_cache.forward(oos)

    # --- hide: stop ---

    kpca_light_cache.reset()
    kpca_light_cache.print_cache()

We can now have a look at the ``'total'`` version. Again, before anything is required, the cache remains empty: nothing
has computed yet.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_total_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='total')

    # --- hide: stop ---

    kpca_total_cache.print_cache()


We now see that everything is saved.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_total_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='total')

    # --- hide: stop ---

    kpca_total_cache.solve()
    kpca_total_cache.forward(oos)
    kpca_total_cache.print_cache()


We can optionally reset the cache again. Nothing is printed: the cache is empty again.

.. exec_code::

    # --- hide: start ---
    import kerch
    import torch

    torch.manual_seed(0)

    sample = torch.randn(5, 3)
    oos = torch.randn(2, 3)

    kpca_total_cache = kerch.level.KPCA(sample=sample,                  # random sample
                                        dim_output=2,                   # we want an output dimension of 2 (the input is 3)
                                        sample_transform=['min'],       # we want the input to be normalized (based on the statistics of the sample)
                                        kernel_transform=['center'],    # we want the kernel to be center
                                        cache_level='total')          

    kpca_total_cache.solve()
    kpca_total_cache.forward(oos)

    # --- hide: stop ---

    kpca_total_cache.reset()
    kpca_total_cache.print_cache()


Managing the Cache
------------------

In the following example, we show how we can add an element to the cache and recover it when called.

.. exec_code::

    import kerch
    import torch
    import time

    class MyCacheExample(kerch.feature.Cache):
        def __init__(self, *args, **kwargs):
            super(MyCacheExample, self).__init__(*args, **kwargs)
            self.big_matrix = kwargs.pop('big_matrix')

        def _compute_qr(self):
            def qr_fun():
                return torch.linalg.qr(self.big_matrix)
            return self._get(key='qr', fun=qr_fun)

        @property
        def Q(self):
            q, r = self._compute_qr()
            return q

        @property
        def R(self):
            q, r = self._compute_qr()
            return r

    # we instantiate our new class
    m = torch.randn(200, 100)
    my_example = MyCacheExample(big_matrix=m)

    # we time our Q property
    start = time.time()
    my_example.Q
    end = time.time()
    print('First access: ' + str(end-start), end='\n\n')

    # we time it again
    start = time.time()
    my_example.Q
    end = time.time()
    print('Second access: ' + str(end-start), end='\n\n')

    # we now have a look at our cache
    my_example.print_cache()




Inheritance Diagram
===================

.. inheritance-diagram::
    kerch.feature.Cache
    :private-bases:
    :top-classes: kerch.feature.Logger torch.nn.Module