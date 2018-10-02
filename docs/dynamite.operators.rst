dynamite\.operators
===================

Single-site operators
---------------------

These operators apply to a single spin, identified by an index passed to their
constructor. The default index is 0. It is implied that they are tensored with
the identity on every other site in the spin chain. Explicitly, a single spin
operator :math:`O` here has the form
:math:`I_0 \otimes I_1 \otimes \ldots \otimes O_{i} \otimes \ldots \otimes I_L`,
where :math:`i` is the index passed in the constructor and :math:`L` is the
length of the spin chain.

.. autofunction:: dynamite.operators.sigmax

.. autofunction:: dynamite.operators.sigmay

.. autofunction:: dynamite.operators.sigmaz

.. autofunction:: dynamite.operators.identity

.. autofunction:: dynamite.operators.zero

Sums and Products
-----------------

.. autofunction:: dynamite.operators.op_sum

.. autofunction:: dynamite.operators.op_product


Translations
^^^^^^^^^^^^

The following two classes translate the given operator ``op`` along the spin chain,
and take the sum or product of the results.

.. autofunction:: dynamite.operators.index_sum

.. autofunction:: dynamite.operators.index_product

Saved Operators
---------------

.. autofunction:: dynamite.operators.load_from_file

.. autofunction:: dynamite.operators.from_bytes

Member Functions
----------------

.. autoclass:: dynamite.operators.Operator
    :members:
