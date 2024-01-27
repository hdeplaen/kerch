Generic Kernels
---------------

.. csv-table::
    :file: generic.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    linear
    rbf
    laplacian
    polynomial
    cosine
    sigmoid
    rff
    nystrom

Network-Based Kernels
---------------------

.. csv-table::
    :file: network.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    explicit_nn
    implicit_nn

Time Kernels
------------

The idea behind time kernels is that time has the same local effect at
all time, or in other words that the kernels are translational invariant. We typically consider the following kernels:

.. csv-table::
    :file: time.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    indicator
    hat
    rbf


Vision Kernels
--------------

.. csv-table::
    :file: vision.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    additive_chi2
    skewed_chi2

Abstract Kernels
----------------

.. csv-table::
    :file: abstract.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    exponential
    explicit
    implicit
    kernel