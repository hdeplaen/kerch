Generic Kernels
---------------

.. csv-table::
    :file: generic/generic.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    generic/linear
    generic/rbf
    generic/laplacian
    generic/polynomial
    generic/cosine
    generic/sigmoid
    generic/rff
    generic/nystrom

Network-Based Kernels
---------------------

.. csv-table::
    :file: network/network.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    network/explicit_nn
    network/implicit_nn

Time Kernels
------------

The idea behind time kernels is that time has the same local effect at
all time, or in other words that the kernels are translational invariant. We typically consider the following kernels:

.. csv-table::
    :file: time/time.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    time/indicator
    time/hat
    generic/rbf


Statistical Kernels
-------------------

.. csv-table::
    :file: statistics/statistics.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    statistics/uniform
    statistics/triangular
    statistics/epanechnikov
    statistics/quartic
    statistics/triweight
    statistics/tricube
    statistics/exponential
    statistics/logistic
    statistics/silverman



Vision Kernels
--------------

.. csv-table::
    :file: vision/vision.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    vision/additive_chi2
    vision/skewed_chi2

Abstract Kernels
----------------

.. csv-table::
    :file: abstract/abstract.csv
    :header-rows: 1

.. toctree::
    :hidden:
    :maxdepth: 3

    abstract/distance
    abstract/explicit
    abstract/implicit
    abstract/kernel