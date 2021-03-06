"""
This module provides the building blocks for Hamiltonians, and
defines their built-in behavior and operations.
"""

import numpy as np

from . import config, validate, msc_tools
from .computations import evolve, eigsolve
from .subspaces import Full
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
        self._mats = {}
        self._msc = None
        self._is_reduced = False
        self._shell = config.shell

        self._subspaces = []

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
        rtn.msc = self.msc.copy()
        rtn.is_reduced = self.is_reduced
        rtn.shell = self.shell

        if self._subspaces:
            for left, right in self.get_subspace_list():
                rtn.add_subspace(left, right)

        rtn.tex = self.tex
        rtn.string = self.string
        rtn.brackets = self.brackets

        return rtn

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
        for left, right in self.get_subspace_list():
            left.L = L
            right.L = L
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
        Get the default left subspace for this operator. This is the subspace most recently
        added with :meth:`Operator.add_subspace`, or config.subspace if
        :meth:`Operator.add_subspace` has not been called.
        """
        space = self.get_subspace_list()[-1][0]
        space.L = self.get_length()
        return space

    @property
    def right_subspace(self):
        """
        Get the default right subspace for this operator. This is the subspace most recently
        added with :meth:`Operator.add_subspace`, or config.subspace if
        :meth:`Operator.add_subspace` has not been called.
        """
        space = self.get_subspace_list()[-1][1]
        space.L = self.get_length()
        return space

    @property
    def subspace(self):
        """
        Get the default subspace for this operator. This is the subspace most recently
        added with :meth:`Operator.add_subspace`, or config.subspace if
        :meth:`Operator.add_subspace` has not been called.
        """
        if self.left_subspace != self.right_subspace:
            raise ValueError("Left and right subspaces are different for this operator. "
                             "use Operator.left_subspace and Operator.right_subspace to "
                             "access them individually.")
        return self.left_subspace

    @subspace.setter
    def subspace(self, value):
        self.add_subspace(value, value)

    def add_subspace(self, left, right=None):
        '''
        Add a pair of subspaces that this operator is compatible with.

        Parameters
        ----------

        left : dynamite.subspaces.Subspace
            A subspace the operator can map to (or multiply from the left)

        right : dynamite.subspaces.Subspace, optional
            A subspace the operator can map from (or multiply to the right). If omitted,
            the left subspace is reused for the right.
        '''
        if right is None:
            right = left

        left = validate.subspace(left)
        right = validate.subspace(right)

        left.L = self.get_length()
        right.L = self.get_length()

        if (left, right) not in self.get_subspace_list():
            self.get_subspace_list().append((left, right))

    def get_subspace_list(self):
        '''
        Return a list of the subspaces that have been registered for this operator.
        '''
        if not self._subspaces:
            if config.subspace is not None:
                self._subspaces = [(config.subspace, config.subspace)]
            else:
                self._subspaces = [(Full(), Full())]

        for left, right in self._subspaces:
            left.L = self.get_length()
            right.L = self.get_length()
        return self._subspaces

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

    def __repr__(self):
        rtn = 'dynamite.Operator on {size} spins:\n'.format(size = self.get_length())
        rtn += self.string
        return rtn

    def table(self):
        '''
        Return a string containing an ASCII table of the coefficients and terms
        that make up this operator.

        The table is generated directly from the MSC representation, so it is
        expanded and simplified to the same form no matter how the operator was
        built.

        Call :meth:`Operator.reduce_msc` first for a more compact table.
        '''
        return msc_tools.table(self.msc, self.get_length())

    def get_latex(self):
        '''
        Return a clean LaTeX representation of the operator.
        '''
        return self.tex.replace('{IDX', '{')

    def _repr_latex_(self):
        return '$' + self.get_latex() + '$'

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

    def get_mat(self, subspaces=None, diag_entries=False):
        """
        Get the PETSc matrix corresponding to this operator, building it if necessary.

        Parameters
        ----------
        subspaces : tuple(Subspace, Subspace), optional
            The subspace pair to get the matrix for. If the matrix is already built for this
            pair, it will be reused. If this option is omitted, the last subspace added with
            :meth:`Operator.add_subspace` will be used, or the Full space by default.

        diag_entries : bool, optional
            Ensure that the sparse matrix has all diagonal elements filled,
            even if they are zero. Some PETSc functions fail if the
            diagonal elements do not exist. Currently a dummy argument; diagonal
            entries are always included.

        Returns
        -------
        petsc4py.PETSc.Mat
            The PETSc matrix corresponding to the operator.
        """
        if subspaces is None:
            subspaces = (self.left_subspace, self.right_subspace)

        if subspaces not in self._mats:
            self.build_mat(subspaces, diag_entries=diag_entries)

        return self._mats[subspaces]

    def build_mat(self, subspaces=None, diag_entries=False):
        """
        Build the PETSc matrix, destroying any matrix that has already been built, and
        store it internally. This function does not return the matrix--see
        :meth:`Operator.get_mat` for that functionality. This function is rarely needed
        by the end user, since it is called automatically whenever the underlying matrix
        needs to be built or rebuilt.
        """

        if subspaces is None:
            subspaces = (self.left_subspace, self.right_subspace)

        if subspaces not in self.get_subspace_list():
            raise ValueError('Attempted to build matrix for a subspace that has not '
                             'been added to the operator.')

        config.initialize()
        from ._backend import bpetsc

        self.reduce_msc()
        term_array = self.msc

        # TODO: keep track of diag_entries
        diag_entries = True
        if term_array[0]['masks'] != 0:
            term_array = np.hstack([np.array([(0,0,0)], dtype=term_array.dtype), term_array])

        masks, indices = np.unique(term_array['masks'], return_index=True)

        # need to add the last index
        mask_offsets = np.ndarray((indices.size+1,), dtype=term_array.dtype['masks'])
        mask_offsets[:-1] = indices
        mask_offsets[-1]  = term_array.shape[0]

        if not msc_tools.is_hermitian(term_array):
            raise ValueError('Building non-Hermitian matrices currently not supported.')

        mat = bpetsc.build_mat(
            L = self.get_length(),
            masks = np.ascontiguousarray(masks),
            mask_offsets = np.ascontiguousarray(mask_offsets),
            signs = np.ascontiguousarray(term_array['signs']),
            coeffs = np.ascontiguousarray(term_array['coeffs']),
            left_type = subspaces[0].to_enum(),
            left_data = subspaces[0].get_cdata(),
            right_type = subspaces[1].to_enum(),
            right_data = subspaces[1].get_cdata(),
            shell = self.shell,
            gpu = config.gpu
        )

        self._mats[subspaces] = mat

    def destroy_mat(self, subspaces=None):
        """
        Destroy the PETSc matrix, freeing the corresponding memory. If the PETSc
        matrix does not exist (has not been built or has already been destroyed),
        the function has no effect.

        Parameters
        ----------
        subspaces : tuple(Subspace, Subspace), optional
            Destroy only the matrix for a particular pair of subspaces.
        """
        if subspaces is not None:
            to_destroy = [subspaces]
        else:
            to_destroy = list(self._mats.keys())

        for k in to_destroy:
            mat = self._mats.pop(k, None)
            if mat is not None:
                mat.destroy()

    def create_states(self):
        '''
        Return a bra and ket compatible with this matrix.

        Returns
        -------
        tuple
            The two states
        '''
        bra = State(self.get_length(), self.left_subspace)
        ket = State(self.get_length(), self.right_subspace)
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

    def to_numpy(self, subspaces=None, sparse=True):
        '''
        Get a SciPy sparse matrix or dense numpy array representing the operator.

        Parameters
        ----------
        subspaces : tuple(Subspace, Subspace), optional
            The subspaces for which to get the matrix. If this option is omitted,
            the last subspace added with :meth:`Operator.add_subspace` will be used,
            or the Full space by default.

        sparse : bool, optional
            Whether to return a sparse matrix or a dense array.

        Returns
        -------
        np.ndarray(dtype = np.complex128)
            The array
        '''

        if subspaces is None:
            subspaces = (self.left_subspace, self.right_subspace)

        ary = msc_tools.msc_to_numpy(self.msc,
                                     (subspaces[0].get_dimension(),
                                      subspaces[1].get_dimension()),
                                     subspaces[0].idx_to_state,
                                     subspaces[1].state_to_idx,
                                     sparse)

        return ary

    def spy(self, subspaces=None, max_size=1024):
        '''
        Use matplotlib to show the nonzero structure of the matrix.

        Parameters
        ----------
        subspaces : tuple(Subspace, Subspace), optional
            The pair of subspaces for which to plot the matrix. Defaults to the most
            recent added with the Operator.add_subspace method, or otherwise
            config.subspace.

        max_size : int, optional
            The maximum matrix dimension for which this function can be called.
            Calling it for too large a matrix will not be informative and probably run
            out of memory, so this is a small safeguard.
        '''
        # TODO: should subspaces really be passed as an argument like that? or should we somehow
        # reference subspaces from the list, like with an index?

        if any(dim > max_size for dim in self.dim):
            raise ValueError('Matrix too big to spy. Either build a smaller operator, or adjust '
                             'the maximum spy size with the argument "max_size"')

        from matplotlib import pyplot as plt
        plt.figure()
        normalized = np.array((self.to_numpy(subspaces=subspaces) != 0).toarray(), dtype = np.float)
        transformed = np.log(normalized + 1E-9)
        plt.imshow(transformed, cmap='Greys')
        plt.show()

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
            return self._vec_mul(x)
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
            self.reduce_msc()
            x.reduce_msc()
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
        rtn.tex = self.with_brackets('tex') + o.with_brackets('tex')
        rtn.brackets = ''
        return rtn

    def dot(self, x, result = None):
        r'''
        Compute the matrix-vector product :math:`\vec{y} = A\vec{x}`

        Parameters
        ----------
        x : dynamite.states.State
            The input state x.

        result : dynamite.states.State, optional
            A state in which to store the result. If omitted, a new State object
            is created.

        Returns
        -------
        dynamite.states.State
            The result
        '''
        right_subspace = x.subspace
        right_match = [(left, right) for left, right in self.get_subspace_list()
                       if right == right_subspace]
        if not right_match:
            raise ValueError('No operator subspace found that matches input vector subspace. '
                             'Try adding the subspace with the Operator.add_subspace method.')

        if result is None:
            if len(right_match) != 1:
                raise ValueError('Ambiguous subspace for result vector. Pass a state '
                                 'with the desired subspace as the "result" option to '
                                 'Operator.dot.')
            left_subspace = right_match[0][0]
            result = State(L=left_subspace.L,
                           subspace=left_subspace)
        else:
            left_subspace = result.subspace

        if (left_subspace, right_subspace) not in right_match:
            raise ValueError('Subspaces of matrix and result vector do not match.')

        self.get_mat(subspaces=(left_subspace, right_subspace)).mult(x.vec, result.vec)
        return result

    def _vec_mul(self, x):
        return self.dot(x)

    def scale(self, x):
        '''
        Scale an operator by a numerical value without making a copy. This is more
        efficient than just doing x*Operator.

        Parameters
        ----------
        x : numeric type
            The coefficient to scale by
        '''
        try:
            self.msc['coeffs'] *= x
        except (ValueError,TypeError):
            raise ValueError('Error attempting to multiply operator by type "%s"' % str(type(x)))

        self.string = '{:.3f}*'.format(x) + self.with_brackets('string')
        self.tex = '{:.3f}*'.format(x) + self.with_brackets('tex')
        self.brackets = ''
        return self

    def _num_mul(self, x):
        rtn = self.copy()
        rtn.scale(x)
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

        sigmax() + sigmay()
        op_sum([sigmax(), sigmay()])
        op_sum(s() for s in [sigmax, sigmay])

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

    iterterms = iter(terms)

    done = False
    for n,t in enumerate(iterterms):
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
        msc_terms.append(msc_tools.msc_sum(t.msc for t in iterterms))

    o.msc = msc_tools.msc_sum(msc_terms)
    o.string = ' + '.join(strings)
    o.tex = ' + '.join(texs)
    o.brackets = '()'
    return o

def op_product(terms):
    """
    A product of several operators. Called in same way as :meth:`op_sum`.
    For example:

    .. code:: python

        >>> sigmax() * sigmay() == op_product([sigmax(), sigmay()])
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
        o.tex = ''.join(texs)
        o.brackets = ''
    else:
        o = identity()

    return o

def index_sum(op, size = None, start = 0, boundary = 'open'):
    """
    Duplicate the operator onto adjacent sites in the spin chain, and sum the resulting
    operators.
    In most cases, ``op`` should have support on site 0 (and possibly others).

    See the examples for more information.

    Parameters
    ----------
    op : Operator
        The operator to translate along the spin chain.

    size : int, optional
        The size of the support of the resulting operator. For open boundary conditions,
        the number of terms in the sum may be smaller than this. If not provided, defaults
        to the value of :attr:`Operator.L`.

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

    rtn.string = 'index_sum(' + op.string + ', sites %d - %d' % (start, stop-1)
    if boundary == 'closed':
        rtn.string += ', wrapped)'
    else:
        rtn.string += ')'

    # add i to the indices for TeX representation
    # TODO: use different letters if we have sum of sums
    sub_tex = op.with_brackets(which = 'tex')
    sub_tex = sub_tex.replace('{IDX', '{IDXi+').replace('{IDXi+0','{IDXi')

    rtn.tex = r'\sum_{i=%d}^{%d}' % (start, stop-1) + sub_tex
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
        The size of the support of the resulting operator. If not provided, defaults
        to the value of :attr:`Operator.L`.

    start : int, optional
        The site for the first operator in the sum.
    """

    if size is None:
        if op.L is None:
            raise ValueError('Must specify index_sum size with either the "size" argument '
                             'or by setting Operator.L (possibly through config.L).')
        else:
            size = op.L

    if size == 0:
        return identity()

    stop = start + size - op.max_spin_idx

    rtn = Operator()
    rtn.msc = msc_tools.msc_product(op.get_shifted_msc(i, wrap_idx = None) for i in range(start, stop))

    rtn.string = 'index_product(' + op.string + ', sites %d - %d)' % (start, stop-1)

    # add i to the indices for TeX representation
    # TODO: use different letters if we have sum of sums
    sub_tex = op.with_brackets(which = 'tex')
    sub_tex = sub_tex.replace('{IDX', '{IDXi+').replace('{IDXi+0','{IDXi')
    rtn.tex = r'\prod_{i=%d}^{%d}' % (start, stop-1) + sub_tex
    rtn.brackets = '[]'

    return rtn

def sigmax(i=0):
    r"""
    The Pauli :math:`\sigma_x` operator on site :math:`i`.
    """
    o = Operator()
    o.msc = [(1<<i, 0, 1)]
    o.tex = r'\sigma^x_{IDX'+str(i)+'}'
    o.string = 'σx'+str(i).join('[]')
    return o

def sigmay(i=0):
    r"""
    The Pauli :math:`\sigma_y` operator on site :math:`i`.
    """
    o = Operator()
    o.msc = [(1<<i, 1<<i, 1j)]
    o.tex = r'\sigma^y_{IDX'+str(i)+'}'
    o.string = 'σy'+str(i).join('[]')
    return o

def sigmaz(i=0):
    r"""
    The Pauli :math:`\sigma_z` operator on site :math:`i`.
    """
    o = Operator()
    o.msc = [(0, 1<<i, 1)]
    o.tex = r'\sigma^z_{IDX'+str(i)+'}'
    o.string = 'σz'+str(i).join('[]')
    return o

def sigma_plus(i=0):
    r"""
    The :math:`\sigma_+ = \sigma_x + i \sigma_y` operator.

    .. note::

        :math:`\sigma_+ = \left( \begin{array}{cc} 0 & 2 \\ 0 & 0 \\ \end{array} \right)`,
        so :math:`S_+ = \left( \begin{array}{cc} 0 & 1 \\ 0 & 0 \\ \end{array} \right) = \frac{1}{2} \sigma_+`
    """
    o = sigmax(i) + 1j*sigmay(i)
    o.tex = r'\sigma^+_{IDX'+str(i)+'}'
    o.string = 'σ+'+str(i).join('[]')
    return o

def sigma_minus(i=0):
    r"""
    The :math:`\sigma_- = \sigma_x - i \sigma_y` operator.

    .. note::

        :math:`\sigma_- = \left( \begin{array}{cc} 0 & 0 \\ 2 & 0 \\ \end{array} \right)`,
        so :math:`S_- = \left( \begin{array}{cc} 0 & 0 \\ 1 & 0 \\ \end{array} \right) = \frac{1}{2} \sigma_-`
    """
    o = sigmax(i) - 1j*sigmay(i)
    o.tex = r'\sigma^-_{IDX'+str(i)+'}'
    o.string = 'σ-'+str(i).join('[]')
    return o

def identity():
    """
    The identity operator.
    """
    o = Operator()
    o.msc = [(0, 0, 1)]
    # TODO: do a fancy double-lined 1?
    o.tex = '1'
    o.string = '1'
    return o

def zero():
    """
    The zero operator---equivalent to a matrix of all zeros.
    """
    o = Operator()
    o.msc = []
    o.tex = '0'
    o.string = '0'
    return o
