
from . import config
config.initialize()

from .operators import sigmax, sigmay, sigmaz, index_product

def commutator(o1, o2):
    """
    The commutator :math:`[O_1,O_2]`.

    Returns
    -------
    dynamite.operators.Operator
        The commutator
    """
    rtn = o1*o2 - o2*o1
    rtn.string = '[%s, %s]' % (o1.string, o2.string)
    rtn.tex = r'\left[ %s, %s \right]' % (o1.tex, o2.tex)
    rtn.brackets = ''
    return rtn

def majorana(idx):
    r"""
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
        rtn = sigmay(b_idx)
    else:
        rtn = sigmax(b_idx)

    if b_idx > 0:
        rtn = index_product(sigmaz(), size=b_idx) * rtn

    rtn.string = 'χ[%d]' % idx
    rtn.tex = r'\chi_{%d}' % idx
    rtn.brackets = ''

    return rtn
