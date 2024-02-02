================
Logging in Kerch
================

Many classes throughout this package display various messages during instantiation and usage. This abstract from
which they descend allows the dislpaying of those messages and control which are printed using the :py:attr:`~kerch.feature.Logger.logging_level`
attribute.

Functionalities
===============

This abstract class has only one purpose: adding a :py:attr:`~kerch.feature.Logger._logger` attribute meant to log various
messages across the package.
Doing it this way allows to get the name of the class instantiated and print more precise messages.


Logging Messages
----------------
The private property ``self._logger`` is an instance of the class
:external+python:py:class:`logging.Logger` of the Python standard library :external+python:doc:`library/logging`.
This allows logging messages by calling, e.g., ``self._logger.debug(message: str)``,
``self._logger.info(message: str)``, ``self._logger.warning(message: str)`` or ``self._logger.error(message: str)``. The
messages will be automatically formatted to reference the appropriate class and give file and line information in debug
mode.

Logging Level
-------------
A particular log level can be set for each instance using the attribute :py:attr:`~kerch.feature.Logger.logging_level`.
If assigned to ``None``,the default logging level will be used.
This value is also set during instantiation using the ``logging_level`` argument. If nothing is specified, ``None`` is
passed leading to the genral default logging level.

The log level always corresponds to an integer. We refer to :external+python:doc:`library/logging` for the
different values.


Default Logging Level
---------------------
This is the default logging level that is used if :py:attr:`~kerch.feature.Logger.logging_level` is never specified (which corresponds to setting it to ``None``).
By default, the logging level is 30, which corresponds to a `warning` level. This default package value can also be changed
and read using the following functions.


.. autofunction:: kerch.set_logging_level

.. autofunction:: kerch.get_logging_level

Abstract Class
==============

.. autoclass:: kerch.feature.Logger
    :members:
    :undoc-members:
    :private-members: _logger
    :show-inheritance:



Example
=======

Default
-------

.. exec_code::

    # --- hide: start ---
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout)
    # --- hide: stop ---

    import kerch

    k = kerch.kernel.RBF()
    k.sample = range(3)
    print(k.K)                                          # first warning (the sigma is defined)


Info
----

.. exec_code::

    # --- hide: start ---
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout)
    # --- hide: stop ---

    import kerch
    import logging

    k = kerch.kernel.RBF(logging_level=logging.INFO)    # first info (no sample initialized yet)
    k.sample = range(3)                                 # second info (the sample is initialized)
    print(k.K)                                          # first warning (the sigma is defined)


Inheritance Diagram
===================

This is a base class that directly inherits from :external+python:py:class:`object`.

