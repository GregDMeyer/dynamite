
from .operators import sigmax, sigmay, sigmaz, index_product


def commutator(op1, op2):
    """
    The commutator :math:`[O_1,O_2]`.

    Returns
    -------
    dynamite.operators.Operator
        The commutator
    """
    rtn = op1*op2 - op2*op1
    rtn._string_rep.string = f'[{op1}, {op2}]'
    rtn._string_rep.tex = r'\left[ %s, %s \right]' % (op1._string_rep.tex, op2._string_rep.tex)
    rtn._string_rep.repr_str = f'commutator({repr(op1)}, {repr(op2)})'
    rtn._string_rep.brackets = ''
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

    rtn._string_rep.string = 'χ[%d]' % idx
    rtn._string_rep.tex = r'\chi_{IDX%d}' % idx
    rtn._string_rep.repr_str = f'majorana({idx})'
    rtn._string_rep.brackets = ''

    return rtn
