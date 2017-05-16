dynamite\.operators
===================

Single-site operators
---------------------

These operators apply to a single spin, identified by an index passed to their constructor. The default index is 0. It is implied that they are tensored with the identity on every other site in the spin chain. Explicitly, a single spin operator :math:`O` here has the form :math:`I_0 \otimes I_1 \otimes \ldots \otimes O_{i} \otimes \ldots \otimes I_L`, where :math:`i` is the ``index`` passed in the constructor and :math:`L` is the length of the spin chain. Note that :math:`L` doesn't have to be set explicitly when the object is constructed.

.. autoclass:: dynamite.operators.Sigmax
    :members:

.. autoclass:: dynamite.operators.Sigmay
    :members:

.. autoclass:: dynamite.operators.Sigmaz
    :members:

.. autoclass:: dynamite.operators.Identity
    :members:

.. autoclass:: dynamite.operators.Zero
    :members:

Sums and Products
-----------------

.. autoclass:: dynamite.operators.Sum
    :members:

.. autoclass:: dynamite.operators.Product
    :members:


Translations
^^^^^^^^^^^^

The following two classes translate the given operator ``op`` along the spin chain,
and take the sum or product of the results.
``op`` should have some non-identity operator on site 0, and possibly include other
sites as well. The operator is translated by some number of sites, ranging from the
argument ``min_i`` to ``max_i`` (inclusive). ``min_i`` defaults to 0, and ``max_i``
defaults to a value such that the operator extends to the end of the spin chain.

.. autoclass:: dynamite.operators.IndexSum
    :members:

.. autoclass:: dynamite.operators.IndexProduct
    :members:

Member Functions
----------------

.. autoclass:: dynamite.operators.Operator
    :members: