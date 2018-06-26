"""
This module provides the building blocks for Hamiltonians, and
defines their built-in behavior and operations.
"""

import numpy as np

from . import config, validate, info, msc_tools
from .computations import evolve, eigsolve
from .subspace import class_to_enum
from .states import State

class Operator:
    """
    A class representing a quantum operator.

    This class generally won't be directly instantiated by the user, but is returned by the
    other functions in this module.
    """

    def __init__(self):

        self._L = config.L
        self._max_spin_idx = None
        self._mat = None
        self._msc = None
        self._is_reduced = False
        self._has_diag_entries = False
        self._shell = config.shell
        self._left_subspace = config.subspace
        self._right_subspace = config.subspace

        self._tex = r'\[\text{operator}\]'
        self._string = '[operator]'
        self._brackets = ''

    def copy(self):
        """
        Return a copy of the operator.
        Copy will not have its PETSc matrix already built,
        even if the operator being copied does.

        Returns
        -------
        Operator
            A copy of the operator
        """
        rtn = Operator()
        rtn.msc = self.msc
        rtn.is_reduced = self.is_reduced
        rtn.shell = self.shell
        rtn.left_subspace = self.left_subspace.copy()
        rtn.right_subspace = self.right_subspace.copy()

        rtn.tex = self.tex
        rtn.string = self.string
        rtn.brackets = self.brackets

        return rtn

    def __del__(self):
        # petsc4py will destroy the matrix object by itself,
        # but for shell matrices the context will not be automatically
        # freed! therefore we need to explicitly call this on object
        # deletion.
        self.destroy_mat()

    ### computations

    def evolve(self, state, t, **kwargs):
        r"""
        Evolve a state under the Hamiltonian. If the Hamiltonian's chain length has not
        been set, attempts to set it based on the state's length.

        This method wraps :meth:`dynamite.computations.evolve` (see that documentation
        for a full description of the method's functionality).

        Parameters
        ----------
        state : dynamite.states.State
            The initial state.

        t : float
            The time :math:`t` for which to evolve the state (can be negative or complex).

        **kwargs :
            Any further keyword arguments are passed to the underlying call to
            :meth:`dynamite.computations.evolve`. See that documentation for a
            detailed description of possible arguments.

        Returns
        -------
        dynamite.states.State
            The result vector :math:`\Psi_f`.
        """

        if self.L is None:
            self.L = state.L

        return evolve(self, state, t, **kwargs)

    def eigsolve(self, **kwargs):
        """
        Find eigenvalues (and eigenvectors if requested) of the Hamiltonian. This class
        method is a wrapper on :meth:`dynamite.computations.eigsolve`. Any keyword
        arguments are passed to that function; see its documentation for details.

        By default, finds one (or possibly a few) eigenvalues with the smallest real
        values (i.e. the ground state).

        .. note:: The spin chain length ``L`` must be set before calling ``eigsolve``.

        Returns
        -------
        numpy.array or tuple(numpy.array, list(dynamite.states.State))
            Either a 1D numpy array of eigenvalues, or a pair containing that array
            and a list of the corresponding eigenvectors.
        """
        return eigsolve(self, **kwargs)

    ### properties

    @property
    def max_spin_idx(self):
        '''
        Read-only property giving the largest spin index on which this operator
        has support.
        '''
        # save this so we don't recompute it every time.
        # cleared when MSC changes

        if self._max_spin_idx is None:
            self._max_spin_idx = msc_tools.max_spin_idx(self.msc)

        return self._max_spin_idx

    @property
    def L(self):
        """
        Property representing the length of the spin chain.
        If L hasn't been set, defaults to the size of support of the operator (from site 0).
        """
        return self._L

    @L.setter
    def L(self, value):
        L = validate.L(value)
        if L < self.max_spin_idx + 1:
            raise ValueError('Cannot set L smaller than one plus the largest spin index'
                             'on which the operator has support (max_spin_idx = %d)' %
                             (self.max_spin_idx))
        self._L = L

    def get_length(self):
        '''
        Returns the length of the spin chain for this operator. It is defined by the
        property :meth:`Operator.L` if it has been set by the user. Otherwise, the
        number of sites on which the operator has support is returned by default.
        '''
        if self.L is None:
            return self.max_spin_idx + 1
        else:
            return self.L

    @property
    def dim(self):
        """
        Read-only attribute returning the dimensions of the matrix.
        """
        return self.left_subspace.get_dimension(), self.right_subspace.get_dimension()

    @property
    def nnz(self):
        """
        The number of nonzero elements per row of the sparse matrix.
        """
        return msc_tools.nnz(self.msc)

    @property
    def msc_size(self):
        """
        The number of elements in the MSC representation of the matrix.
        """
        return len(self.msc)

    @property
    def density(self):
        """
        The density of the sparse matrix---that is, the number of non-zero
        elements per row divided by the length of a row.

        .. note::
            This quantity is not always well defined when using a subspace, since
            it can vary by row. In that case, the returned quantity will be an upper bound.
        """
        return self.nnz/self.dim[1]

    @property
    def shell(self):
        """
        Switch whether to use shell matrices or not. For a description of shell
        matrices and their benefits, see the documentation.

        .. note::
            Changing this value after the matrix has been built will invoke a call
            to :meth:`Operator.destroy_mat`.
        """
        return self._shell

    @shell.setter
    def shell(self,value):
        value = validate.shell(value)
        if value != self._shell:
            self.destroy_mat()
        self._shell = value

    @property
    def left_subspace(self):
        """
        The subspace of "bra" states--in other words, states that
        could be multiplied on the left against the matrix.
        """
        # always make sure that the subspace associated with this operator
        # has the correct dimensions
        self._left_subspace.L = self.get_length()
        return self._left_subspace

    @property
    def right_subspace(self):
        """
        The subspace of "ket" states--those that can be multiplied
        on the right against the matrix.
        """
        self._right_subspace.L = self.get_length()
        return self._right_subspace

    @property
    def subspace(self):
        """
        When the left and right subspaces are the same, this property
        gets or sets both at the same time. If they are different, it
        raises an error.
        """
        if self.left_subspace != self.right_subspace:
            raise ValueError('Left subspace and right subspace not the same, '
                             'use left_subspace and right_subspace to access each.')

        return self.left_subspace

    @left_subspace.setter
    def left_subspace(self, s):
        validate.subspace(s)
        if s != self.left_subspace:
            self.destroy_mat()
        self._left_subspace = s

    @right_subspace.setter
    def right_subspace(self, s):
        validate.subspace(s)
        if s != self.right_subspace:
            self.destroy_mat()
        self._right_subspace = s

    @subspace.setter
    def subspace(self, s):
        self.left_subspace = s
        self.right_subspace = s

    ### text representations

    # TODO: perhaps encapsulate the string/tex methods into their own class

    @property
    def string(self):
        '''
        A text string that will be used to represent the object in printed expressions.
        '''
        return self._string

    @string.setter
    def string(self, value):
        self._string = value

    @property
    def tex(self):
        '''
        A LaTeX expression corresponding to the object. Can be set to any valid TeX.
        '''
        return self._tex

    @tex.setter
    def tex(self, value):
        self._tex = value

    @property
    def brackets(self):
        '''
        Which kind of brackets to surround the expression with. Options are
        ``'()'``, ``'[]'``, or ``''``, where the empty string means no brackets.
        '''
        return self._brackets

    @brackets.setter
    def brackets(self, value):
        value = validate.brackets(value)
        self._brackets = value

    @classmethod
    def _with_brackets(cls, string, brackets, tex=False):
        '''
        Put the given brackets around the string. If tex = True, the brackets
        have \left and \right appended to them.

        Parameters
        ----------
        string : str
            The string to put brackets around

        brackets : str
            The set of brackets. Should be either ``'[]'``, ``'()'``, or ``''``
            for no brackets.

        tex : bool, optional
            Whether to prepend ``\left`` and ``\right`` to the brackets.

        Returns
        -------
        str
            The result
        '''
        if not brackets:
            return string
        if tex:
            brackets = [x+y for x,y in zip([r'\left',r'\right'], brackets)]
        return string.join(brackets)

    def with_brackets(self, which):
        '''
        Return a string or tex representation of the object, surrounded by brackets
        if necessary. Useful for building larger expressions.

        Parameters
        ----------

        which : str
            Whether to return a normal string or tex. Options are ``'string'`` or ``'tex'``.
        '''
        if which == 'tex':
            strng = self.tex
        elif which == 'string':
            strng = self.string
        else:
            raise ValueError("which must be either 'string' or 'tex'.")

        return self._with_brackets(strng, self._brackets, which == 'tex')

    def __str__(self):
        return self.string

    # TODO: __repr__

    def table(self):
        '''
        Return a string containing an ASCII table of the coefficients and terms
        that make up this operator.

        The table is generated directly from the MSC representation, so it is
        expanded and simplified to the same form no matter how the operator was
        built.

        Call :meth:`Operator.reduce_msc` first for a more compact table.

        [This function is not yet implemented]
        '''
        raise NotImplementedError

    def _repr_latex_(self):
        return '$' + self.tex + '$'

    ### save to disk

    def serialize(self):
        '''
        Serialize the operator's MSC representation into a string of bytes.
        The byte string ONLY contains the MSC representation and the spin chain
        length. It does not save any other information, such as subspaces etc.

        Returns
        -------
        bytes
            The byte string containing the serialized object.

        '''
        return msc_tools.serialize(self.msc)

    def save(self, filename):
        """
        Save the MSC representation of the operator to disk.
        Can be loaded again through :class:`Load`.

        .. note::
            If one calls this method in parallel, one MUST call :meth:`dynamite.config.initialize`
            first, or all processes will try to simultaneously write to the same file!

        Parameters
        ----------
        filename : str
            The path to the file to save the operator in.
        """

        if config.initialized:
            from petsc4py import PETSc
            do_save = PETSc.COMM_WORLD.rank == 0
        else:
            # this should be the case when not running under MPI
            do_save = True

        # only process 0 should save
        if do_save:
            with open(filename, mode='wb') as f:
                f.write(self.serialize())

        if config.initialized:
            PETSc.COMM_WORLD.barrier()

    ### interface with PETSc

    def get_mat(self, diag_entries=False):
        """
        Get the PETSc matrix corresponding to this operator, building it if necessary.

        Parameters
        ----------
        diag_entries : bool
            Ensure that the sparse matrix has all diagonal elements filled,
            even if they are zero. Some PETSc functions fail if the
            diagonal elements do not exist.

        Returns
        -------
        petsc4py.PETSc.Mat
            The PETSc matrix corresponding to the operator.
        """
        if self._mat is None or (diag_entries and not self._has_diag_entries):
            self.build_mat(diag_entries=diag_entries)
        return self._mat

    def build_mat(self, diag_entries=False):
        """
        Build the PETSc matrix, destroying any matrix that has already been built, and
        store it internally. This function does not return the matrix--see
        :meth:`Operator.get_mat` for that functionality. This function is rarely needed
        by the end user, since it is called automatically whenever the underlying matrix
        needs to be built or rebuilt.
        """

        # we wait to import until we need to, avoiding initialization
        from ._backend import bpetsc

        if self.get_length() > self.max_spin_idx:
            info.write(1, 'Trivial identity operators at end of chain--length L could be smaller.')

        self.destroy_mat()
        self.reduce_msc()
        term_array = self.msc

        self._has_diag_entries = term_array[0]['masks'] == 0
        if diag_entries and not self._has_diag_entries:
            term_array = np.hstack([np.array([(0,0,0)], dtype=term_array.dtype), term_array])
            self._has_diag_entries = True

        self._mat = bpetsc.build_mat(L = self.get_length(),
                                     masks = np.ascontiguousarray(term_array['masks']),
                                     signs = np.ascontiguousarray(term_array['signs']),
                                     coeffs = np.ascontiguousarray(term_array['coeffs']),
                                     left_type = class_to_enum(type(self.left_subspace)),
                                     left_space = self.left_subspace.space,
                                     right_type = class_to_enum(type(self.right_subspace)),
                                     right_space = self.right_subspace.space,
                                     shell = bool(self.shell),
                                     gpu = self.shell == 'gpu')

    def destroy_mat(self):
        """
        Destroy the PETSc matrix, freeing the corresponding memory. If the PETSc
        matrix does not exist (has not been built or has already been destroyed),
        the function has no effect.
        """

        if self._mat is None:
            return

        # TODO: see if there is a way that I can add destroying the shell
        # context to the __del__ method for the matrix
        if self._shell:
            config.initialize()
            from ._backend import bpetsc
            bpetsc.destroy_shell_context(self._mat)

        self._mat.destroy()
        self._mat = None

    def create_states(self):
        '''
        Return a bra and ket compatible with this matrix.

        Returns
        -------
        tuple
            The two states
        '''
        bra = State(self.L, self.left_subspace)
        ket = State(self.L, self.right_subspace)
        return (bra, ket)

    ### mask, sign, coefficient representation of operators

    @property
    def msc(self):
        '''
        The (mask, sign, coefficient) representation of the operator. This
        representation is used internally by dynamite.
        '''
        return self._msc

    @msc.setter
    def msc(self, value):
        value = validate.msc(value)
        self._max_spin_idx = None
        self.is_reduced = False
        self._msc = value

    def reduce_msc(self):
        '''
        Combine and sort terms in the MSC representation, compressing it and
        preparing it for use in the backend.
        '''
        self.msc = msc_tools.combine_and_sort(self.msc)
        self.is_reduced = True

    @property
    def is_reduced(self):
        '''
        Whether :meth:`Operators.reduce_msc` has been called. Can also be set manually to avoid
        calling that function, if you are sure that the terms are sorted already.
        '''
        return self._is_reduced

    @is_reduced.setter
    def is_reduced(self, value):
        self._is_reduced = value

    def get_shifted_msc(self, shift, wrap_idx = None):
        '''
        Get the MSC representation of the operator, with all terms translated along
        the spin chain (away from zero) by ``shift`` spins.

        Parameters
        ----------
        shift : int
            Shift the whole operator along the spin chain by ``shift`` spins.

        wrap : bool
            The site at which to begin wrapping around to the beginning of the spin chain.
            e.g. takes a site index ``i`` to ``i % wrap_idx``. If ``None``, do not wrap.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the representation.
        '''
        return msc_tools.shift(self.msc, shift, wrap_idx)

    ### interface to numpy

    def to_numpy(self, sparse = True):
        '''
        Get a SciPy sparse matrix or dense numpy array representing the operator.

        Parameters
        ----------
        sparse : bool, optional
            Whether to return a sparse matrix or a dense array.

        Returns
        -------
        np.ndarray(dtype = np.complex128)
            The array
        '''

        ary = msc_tools.msc_to_numpy(self.msc,
                                     (self.left_subspace.get_dimension(),
                                      self.right_subspace.get_dimension()),
                                     self.left_subspace.idx_to_state,
                                     self.right_subspace.state_to_idx,
                                     sparse)

        return ary

    ### unary and binary operations

    def __add__(self, x):
        if not isinstance(x, Operator):
            x = x*identity()
        return self._op_add(x)

    def __radd__(self,x):
        if not isinstance(x, Operator):
            x = x*identity()
        return x + self

    def __sub__(self, x):
        return self + -x

    def __neg__(self):
        return -1*self

    def __mul__(self, x):
        if isinstance(x, Operator):
            return self._op_mul(x)
        elif isinstance(x, State):
            # TODO: check that the subspaces match?
            return self.get_mat() * x.vec
        else:
            return self._num_mul(x)

    def __rmul__(self, x):
        if isinstance(x, State):
            return TypeError('Left vector-matrix multiplication not currently '
                             'supported.')
        else:
            return self._num_mul(x)

    def __eq__(self, x):
        if isinstance(x, Operator):
            return np.array_equal(self.msc, x.msc)
        else:
            raise TypeError('Equality not supported for types %s and %s'
                            % (str(type(self)), str(type(x))))

    def _op_add(self, o):
        rtn = self.copy()
        rtn.msc = msc_tools.msc_sum([self.msc, o.msc])
        rtn.tex = self.tex + ' + ' + o.tex
        rtn.string = self.string + ' + ' + o.string
        rtn.brackets = '()'
        return rtn

    def _op_mul(self, o):
        rtn = self.copy()
        rtn.msc = msc_tools.msc_product([self.msc, o.msc])
        rtn.string = self.with_brackets('string') + '*' + o.with_brackets('string')
        rtn.tex = self.with_brackets('tex') + '*' + o.with_brackets('tex')
        rtn.brackets = ''
        return rtn

    def _num_mul(self, x):
        rtn = self.copy()
        rtn.msc['coeffs'] *= x
        rtn.string = '{:.3f}*'.format(x) + self.with_brackets('string')
        rtn.tex = '{:.3f}*'.format(x) + self.with_brackets('tex')
        rtn.brackets = ''
        return rtn

def load_from_file(filename):
    '''
    Load the operator in file ``filename`` and return the corresponding object.

    Parameters
    ----------
    filename : str
        The path of the file to load.

    Returns
    -------
    dynamite.operators.Load
        The operator as a dynamite object.
    '''
    with open(filename, 'rb') as f:
        bytestring = f.read()
        op = from_bytes(bytestring)
    return op

def from_bytes(data):
    """
    Load operator from a byte string generated with the :meth:`Operator.serialize`
    method.

    Parameters
    ----------
    data : bytes
        The byte string containing the serialized object.

    Returns
    -------
    Operator
        The operator.
    """
    o = Operator()
    msc = msc_tools.deserialize(data)
    o.msc = msc
    o.string = '[operator from bytes]'
    o.tex = r'\left[\text{operator from bytes}\right]'
    return o

def op_sum(terms, nshow = 3):
    r"""
    A sum of several operators. This object can be used in a couple ways.
    All of the following return the exact same object,
    :math:`\sigma^x_0 + \sigma^y_0`\:

    .. code:: python

        Sigmax() + Sigmay()
        op_sum([Sigmax(),Sigmay()])
        op_sum(s() for s in [Sigmax,Sigmay])

    Parameters
    ----------
    terms : list
        A list of operators to sum

    nshow : int, optional
        The number of terms to show in the string representations before adding
        an ellipsis.
    """

    o = Operator()
    msc_terms = []
    strings = []
    texs = []

    done = False
    for n,t in enumerate(terms):
        msc_terms.append(t.msc)
        strings.append(t.string)
        texs.append(t.tex)
        if n >= nshow:
            break
    else:
        done = True

    if not done:
        strings[-1] = '...'
        texs[-1] = r'\cdots'
        msc_terms.append(msc_tools.msc_sum(terms))

    o.msc = msc_tools.msc_sum(msc_terms)
    o.string = ' + '.join(strings)
    o.tex = ' + '.join(texs)
    o.brackets = '()'
    return o

def op_product(terms):
    """
    A product of several operators. Called in same way as :class:`Sum`.
    For example:

    .. code:: python

        >>> Sigmax() * Sigmay() == Product([Sigmax(),Sigmay()])
        True

    Parameters
    ----------
    terms : list
        A list of operators to multiply
    """

    # from a practical standpoint, there doesn't seem to ever be a use case
    # for taking the product of a huge number of terms. So we assume the number
    # of terms is O(1) in this implementation.

    msc_terms = []
    strings = []
    texs = []
    for t in terms:
        msc_terms.append(t.msc)
        strings.append(t.with_brackets(which='string'))
        texs.append(t.with_brackets(which='tex'))

    if msc_terms:
        o = Operator()
        o.msc = msc_tools.msc_product(msc_terms)
        o.string = '*'.join(strings)
        o.tex = '*'.join(texs)
        o.brackets = ''
    else:
        o = identity()

    return o

def index_sum(op, size = None, start = 0, boundary = 'open'):
    """
    Duplicate the operator onto adjacent sites in the spin chain, and sum the resulting
    operators.
    In most cases, ``op`` should have support on site 0 (and possibly others).

    For illustrative examples, see the Examples pages.

    Parameters
    ----------
    op : Operator
        The operator to translate along the spin chain.

    size : int, optional
        The size of the support of the resulting operator. For open boundary conditions,
        the number of terms in the sum may be smaller than this. If not provided, defaults
        to the value of :meth:`Operator.L`.

    start : int, optional
        The site for the first operator in the sum.

    boundary : str, optional
        Whether to use 'open' or 'closed' boundary conditions. When ``op`` has support
        on more than one site, this determines whether the last few terms of the sum should
        wrap around to the beginning of the spin chain.
    """

    if size is None:
        if op.L is None:
            raise ValueError('Must specify index_sum size with either the "size" argument '
                             'or by setting Operator.L (possibly through config.L).')
        else:
            size = op.L

    if boundary == 'open':
        stop = start + size - op.max_spin_idx
        if stop <= start:
            raise ValueError("requested size %d for sum operator's support smaller than "
                             "summand's support %d; impossible to satisfy" % \
                             (size, op.max_spin_idx))
        wrap_idx = None

    elif boundary == 'closed':
        stop = start + size
        wrap_idx = stop
        if start != 0:
            raise ValueError('cannot set start != 0 for closed boundary conditions.')

    else:
        raise ValueError("invalid value for argument 'boundary' (can be 'open' or 'closed')")

    rtn = Operator()
    rtn.msc = msc_tools.msc_sum(op.get_shifted_msc(i, wrap_idx) for i in range(start, stop))

    rtn.string = 'index_sum(' + op.string + ', sites %d - %d)' % (start, stop-1)
    # TODO: make tex prettier by substituting i for the indices
    rtn.tex = r'\sum_{i=%d}^{%d}' % (start, stop-1) + op.with_brackets(which = 'tex') + '_{i}'
    rtn.brackets = '[]'

    return rtn

def index_product(op, size = None, start = 0):
    """
    Duplicate the operator onto adjacent sites in the spin chain, and multiply the
    resulting operators together.
    In most cases, ``op`` should have support on site 0 (and possibly others).

    Parameters
    ----------
    op : Operator
        The operator to translate along the spin chain.

    size : int, optional
        The size of the support of the resulting operator. If omitted, defaults to
        the spin chain length set by :meth:`Operator.L` or :meth:`dynamite.config.L`.

    start : int, optional
        The site for the first operator in the sum.
    """

    if size is None:
        if op.L is None:
            raise ValueError('Must specify index_sum size with either the "size" argument '
                             'or by setting Operator.L (possibly through config.L).')
        else:
            size = op.L

    stop = start + size - op.max_spin_idx

    rtn = Operator()
    rtn.msc = msc_tools.msc_product(op.get_shifted_msc(i, wrap_idx = None) for i in range(start, stop))

    rtn.string = 'index_product(' + op.string + ', sites %d - %d)' % (start, stop-1)
    # TODO: make tex prettier by substituting i for the indices
    rtn.tex = r'\prod_{i=%d}^{%d}' % (start, stop-1) + op.with_brackets(which = 'tex') + '_{i}'
    rtn.brackets = '[]'

    return rtn

def sigmax(i=0):
    r"""
    The Pauli :math:`\sigma_x` operator on site :math:`i`.
    """
    o = Operator()
    o.msc = [(1<<i, 0, 1)]
    o.tex = r'\sigma^x_{'+str(i)+'}'
    o.string = 'σx'+str(i).join('[]')
    return o

def sigmay(i=0):
    r"""
    The Pauli :math:`\sigma_y` operator on site :math:`i`.
    """
    o = Operator()
    o.msc = [(1<<i, 1<<i, 1j)]
    o.tex = r'\sigma^y_{'+str(i)+'}'
    o.string = 'σy'+str(i).join('[]')
    return o

def sigmaz(i=0):
    r"""
    The Pauli :math:`\sigma_z` operator on site :math:`i`.
    """
    o = Operator()
    o.msc = [(0, 1<<i, 1)]
    o.tex = r'\sigma^z_{'+str(i)+'}'
    o.string = 'σz'+str(i).join('[]')
    return o

def identity():
    """
    The identity operator. Since it is tensored with identities on all
    the rest of the sites, the ``index`` argument has no effect.
    """
    o = Operator()
    o.msc = [(0, 0, 1)]
    # TODO: do a fancy double-lined 1?
    o.tex = '1'
    o.string = '1'
    return o

def zero():
    """
    The zero operator---equivalent to a matrix of all zeros of dimension :math:`2^L`.
    Like for the identity, the ``index`` argument has no effect.
    """
    o = Operator()
    o.msc = []
    o.tex = '0'
    o.string = '0'
    return o
