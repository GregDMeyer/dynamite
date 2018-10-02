dynamite.config
===============

The :class:`dynamite._Config` class is instantiated by dynamite
as ``dynamite.config``. This is the object on which one should call
these functions. For example:

.. code:: python

    from dynamite import config

    config.initialize(['-mfn_ncv','20'])
    config.L = 24

.. autoclass:: dynamite._Config
    :members:
