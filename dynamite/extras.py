
from . import config
config.initialize()

from .operators import Sigmax,Sigmay,Sigmaz,IndexProduct

def commutator(o1,o2):
    """
    The commutator :math:`[O_1,O_2]`.

    Returns
    -------
    dynamite.operators.Operator
        The commutator
    """
    return o1*o2 - o2*o1

def Majorana(idx):
    """
    A function generating an operator that represents a
    Majorana fermion as a boundary in a spin chain.

    The boundary is at index :code:`b_idx = floor(idx/2) + 1`.
    The operator consists of the tensor product of :math:`\sigma_z`
    operators up to spin ``b_idx - 1``, and then on spin ``b_idx``
    a :math:`\sigma_x` operator if ``idx`` is even or a :math:`\sigma_y`
    operator if ``idx`` is odd.

    Parameters
    ----------
    idx : int
        The index of the Majorana

    Returns
    -------
    dynamite.operators.Operator
        The Majorana of index ``idx``
    """

    b_idx = idx//2
    parity = idx % 2
    if parity:
        m = Sigmay(b_idx)
    else:
        m = Sigmax(b_idx)

    if b_idx > 0:
        m = IndexProduct( Sigmaz(),max_i=b_idx-1) * m

    return m
