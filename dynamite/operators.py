
"""
This module provides the building blocks for Hamiltonians, and
defines their built-in behavior and operations.
"""

# defines the order the classes appear in the documentation
__all__ = ['Sigmax',
           'Sigmay',
           'Sigmaz',
           'Identity',
           'Zero',
           'Sum',
           'Product',
           'IndexSum',
           'IndexProduct',
           'Load']

import numpy as np
from copy import deepcopy

from . import config, validate, info
from .computations import evolve, eigsolve
from .subspace import class_to_enum
from .states import State
from ._imports import get_import
from . import _utils
from ._utils import condense_terms,coeff_to_str,MSC_matrix_product

# this will be serial_backend eventually
from .backend import backend

class Operator:
    """
    Operator is the base class for all other operator classes.
    This class should not be constructed explicitly by the user,
    however here are defined some important member functions all
    operators will have (see below).

    Parameters
    ----------
    L : int, optional
        The length of the spin chain. Can be set later with ``op.L = L``.
    """

    def __init__(self,L=None):

        if L is None:
            L = config.L

        self._L = None
        self._mat = None
        self._MSC = None
        self._diag_entries = False
        self._shell = config.shell
        self._left_subspace = config.subspace
        self._right_subspace = config.subspace

        self.min_length = 0
        self.needs_parens = False
        self.coeff = 1
        self.L = L

    ### computations

    def evolve(self,state,t,**kwargs):
        """
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

        return evolve(self,state,t,**kwargs)

    def eigsolve(self,**kwargs):
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
        return eigsolve(self,**kwargs)

    ### properties

    @property
    def L(self):
        """
        Property defining the length of the spin chain.
        If the length hasn't been set yet, ``L`` is ``None``.
        """
        return self._L

    @L.setter
    def L(self,value):

        if value is not None:
            value = validate.L(value)

            if value < self.min_length:
                raise ValueError('Length %d too short--non-identity'
                                 'operator on index %d' % (value,self.min_length-1))

        self.destroy_mat()
        self._MSC = None

        self._L = value

        self.left_subspace.L = value
        self.right_subspace.L = value

        # propagate L down the tree of operators
        self._prop_L(value)

    def _prop_L(self,L):
        raise NotImplementedError()

    @property
    def dim(self):
        """
        Read-only attribute returning the dimension of the matrix,
        or ``None`` if ``L`` is ``None``.
        """
        if self.L is None:
            return None
        else:
            return self.left_subspace.get_size(), self.right_subspace.get_size()

    @property
    def nnz(self):
        """
        The number of nonzero elements per row of the sparse matrix.
        """
        t = self.get_MSC()
        return len(np.unique(t['masks']))

    @property
    def MSC_size(self):
        """
        The number of elements in the MSC representation of the matrix.
        """
        return len(self.get_MSC())

    @property
    def density(self):
        """
        The density of the sparse matrix---that is, the number of non-zero
        elements per row divided by the length of a row.
        """
        return self.nnz/self.dim[0]

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
        if value != self._shell:
            self.destroy_mat()
        self._shell = value

    @property
    def use_shell(self):
        raise TypeError('Operator.use_shell is deprecated, use Operator.shell instead.')

    @use_shell.setter
    def use_shell(self,value):
        raise TypeError('Operator.use_shell is deprecated, use Operator.shell instead.')

    @property
    def left_subspace(self):
        """
        The subspace of "bra" states--in other words, states that
        could be multiplied on the left against the matrix.
        """
        return self._left_subspace

    @property
    def right_subspace(self):
        """
        The subspace of "ket" states--those that can be multiplied
        on the right against the matrix.
        """
        return self._right_subspace

    @property
    def subspace(self):
        """
        When the left and right subspaces are the same, this property
        gets or sets both at the same time. If they are different, it
        raises an error.
        """
        if self._left_subspace != self._right_subspace:
            raise ValueError('Left subspace and right subspace not the same, '
                             'use left_subspace and right_subspace to access each.')

        return self._left_subspace

    @left_subspace.setter
    def left_subspace(self,s):
        validate.subspace(s)
        s.L = self.L
        s.update_operator('left',self)
        self._left_subspace = s

    @right_subspace.setter
    def right_subspace(self,s):
        validate.subspace(s)
        s.L = self.L
        s.update_operator('right',self)
        self._right_subspace = s

    @subspace.setter
    def subspace(self,s):
        self.left_subspace = s
        self.right_subspace = s

    ### copy

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

        # the only thing we really don't want to copy is the PETSc matrix in the
        # backend. so we just temporarily make that not part of the operator!

        tmp_mat = self._mat
        self._mat = None
        c = deepcopy(self)
        self._mat = tmp_mat
        return c

    ### save to disk

    def save(self,fout):
        """
        Save the MSC representation of the operator to disk.
        Can be loaded again through :class:`Load`.

        .. note::
            If one calls this method in parallel, one MUST call :meth:`dynamite.config.initialize`
            first, or all processes will try to simultaneously write to the same file!

        Parameters
        ----------
        fout : str
            The path to the file to save the operator in.
        """

        if config.initialized:
            PETSc = get_import('petsc4py.PETSc')
            do_save = PETSc.COMM_WORLD.rank == 0
        else:
            # this should be the case when not running under MPI
            do_save = True

        # only process 0 should save
        if do_save:

            # The file format is:
            # L,nterms,masks,signs,coefficients
            # where each is just a binary blob, one after the other.

            # values are saved in big-endian format, to be compatible with PETSc defaults

            if self.L is None:
                raise ValueError('L must be set before saving to disk.')

            msc = self.get_MSC()

            with open(fout,mode='wb') as f:

                # cast it to the type that C will be looking for
                int_t = msc.dtype[0].newbyteorder('>')
                complex_t = msc.dtype[2].newbyteorder('>')

                L = np.array(self.L,dtype=int_t)
                f.write(L.tobytes())

                # write out the length of the MSC representation
                size = np.array(msc.size,dtype=int_t)
                f.write(size.tobytes())

                f.write(msc['masks'].astype(int_t,casting='equiv',copy=False).tobytes())
                f.write(msc['signs'].astype(int_t,casting='equiv',copy=False).tobytes())
                f.write(msc['coeffs'].astype(complex_t,casting='equiv',copy=False).tobytes())

        if config.initialized:
            PETSc.COMM_WORLD.barrier()

    ### interface with PETSc

    def get_mat(self,diag_entries=False):
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
        if self._mat is None or (diag_entries and not self._diag_entries):
            self.build_mat(diag_entries=diag_entries)
        return self._mat

    def destroy_mat(self):
        """
        Destroy the PETSc matrix associated with the Hamiltonian, and free the
        associated memory. If the PETSc matrix does not exist (has not been built
        or has already been destroyed), the function has no effect.
        """

        if self._mat is None:
            return

        # TODO: see if there is a way that I can add destroying the shell
        # context to the __del__ method for the matrix
        if self._shell:
            backend = get_import('backend')
            backend.destroy_shell_context(self._mat)

        self._mat.destroy()
        self._mat = None

    def build_mat(self,diag_entries=False):
        """
        Build the PETSc matrix, destroying any matrix that has already been built, and
        store it internally. This function does not return the matrix--see
        :meth:`Operator.get_mat` for that functionality. This function is rarely needed
        by the end user, since it is called automatically whenever the underlying matrix
        needs to be built or rebuilt.
        """

        if self.L is None:
            raise ValueError('Set L before building matrix.')

        if self.L > self.min_length:
            info.write(1,'Trivial identity operators at end of chain--length L could be smaller.')

        # TODO: will be mpi_backend eventually
        backend = get_import('backend')

        self.destroy_mat()

        term_array = self.get_MSC()

        if diag_entries and not np.any(term_array['masks'] == 0):
            term_array = np.hstack([np.array([(0,0,0)],dtype=term_array.dtype),term_array])

        if not np.any(term_array['masks'] == 0):
            self._diag_entries = False
        else:
            self._diag_entries = True

        # TODO: clean up these arguments into a dictionary or similar
        self._mat = backend.build_mat(self.L,
                                      np.ascontiguousarray(term_array['masks']),
                                      np.ascontiguousarray(term_array['signs']),
                                      np.ascontiguousarray(term_array['coeffs']),
                                      class_to_enum(type(self.left_subspace)),
                                      self.left_subspace.space,
                                      class_to_enum(type(self.right_subspace)),
                                      self.right_subspace.space,
                                      bool(self.shell),
                                      self.shell == 'gpu')

    ### LaTeX representation of operators

    def build_tex(self,signs='-',request_parens=False):
        """
        Output LaTeX of a symbolic representation of the Hamiltonian.
        The arguments are probably only useful if integrating the output
        into some larger LaTeX expression, but they are used internally
        and could be useful for somebody so I expose them in the API.

        Parameters
        ----------
        signs : str
            The possible signs to include at the front of the TeX. Possible
            values are ``'+-'``, ``'-'``, and ``''``. If the coefficient on
            the expression is positive, a + sign will only be included if +
            is included in the ``signs`` argument. Same with -, if the
            coefficient is negative.

        request_parens : bool
            If set to ``True``, the returned expression will include surrounding
            parentheses if they are needed (e.g. not if the result is a single
            symbol that never would need parentheses around it).
        """
        return self._build_tex(signs,request_parens)

    def _build_tex(self,signs='-',request_parens=False):
        raise NotImplementedError()

    def _repr_latex_(self):

        # don't print if we aren't process 0
        # this causes there to still be some awkward white space,
        # but at least it doesn't print the tex a zillion times
        if config.initialized:
            PETSc = get_import('petsc4py.PETSc')
            if PETSc.COMM_WORLD.getRank() != 0:
                return ''

        return '$' + self.build_tex(signs='-') + '$'

    def _get_index_set(self):
        # Get the set of indices in the TeX representation of the object
        raise NotImplementedError()

    def _replace_index(self,ind,rep):
        # Replace an index by some value in the TeX representation
        raise NotImplementedError()

    ### mask, sign, coefficient representation of operators

    def get_MSC(self,shift_index=None,wrap=False):
        """
        Get the representation of the operator in the (mask, sign, coefficient)
        format used internally by :mod:`dynamite`. The representation is saved
        internally, so that if it takes a while to build it can be accessed quickly
        next time. It is deleted when the PETSc matrix is built, however, to free
        up some memory.

        Parameters
        ----------
        shift_index : int
            Shift the whole operator along the spin chain by ``shift_index`` spins.

        wrap : bool
            Whether to wrap around if requesting a ``shift_index`` greater than the length
            of the spin chain (i.e., take ``shift_index`` to ``shift_index % L``).

        Returns
        -------
        numpy.ndarray
            A numpy array containing the representation.
        """

        if self._MSC is None:
            self._MSC = self._get_MSC()

        if shift_index is None:
            return self._MSC
        else:
            # check that the shift_index is valid
            if not wrap and self.L is not None and self.min_length + shift_index > self.L:
                raise ValueError('Shift of %d would put operators past end of spin chain.'%
                                 shift_index)

            msc = self._MSC.copy()

            msc['masks'] <<= shift_index
            msc['signs'] <<= shift_index

            if wrap:
                mask = (-1) << self.L

                for v in [msc['masks'],msc['signs']]:

                    # find the bits that need to wrap around
                    overflow = v & mask

                    # wrap them to index 0
                    overflow >>= self.L

                    # recombine them with the ones that didn't get wrapped
                    v |= overflow

                    # shave off the extras that go past L
                    v &= ~mask

            return msc

    def _get_MSC(self):
        raise NotImplementedError()

    ### interface to numpy

    @classmethod
    def _MSC_to_numpy(cls, MSC, dims, idx_to_state = None, state_to_idx = None):
        '''
        Build a NumPy array from an MSC array. This method isolates to_numpy
        from the rest of the class for testing. It also defines the MSC
        representation.

        Parameters
        ----------

        MSC : np.ndarray(dtype = MSC_dtype)
            An MSC array.

        dims : (int, int)
            The dimensions (M, N) of the matrix.

        idx_to_state : function(int), optional
            If working in a subspace, a function to map indices to states for
            the *left* subspace.

        state_to_idx : function(int), optional
            If working in a subspace, a function to map states to indices for
            the *right* subspace.

        Returns
        -------

        np.ndarray(dtype = np.complex128)
            A 2-D NumPy array which stores the matrix.
        '''

        ary = np.zeros(dims, dtype = np.complex128)

        # if these aren't supplied, they are the identity
        if idx_to_state is None:
            idx_to_state = lambda x: x

        if state_to_idx is None:
            state_to_idx = lambda x: x

        for idx in range(dims[0]):
            bra = idx_to_state(idx)
            for m,s,c in MSC:
                ket = m ^ bra
                ridx = state_to_idx(ket)
                if ridx is not None: # otherwise we went out of the subspace
                    sign = 1 - 2*(_utils.popcount(s & ket) % 2)
                    ary[idx, ridx] += sign * c

        return ary

    def to_numpy(self):
        '''
        Get a dense NumPy array representing the operator.

        Returns
        -------

        np.ndarray(dtype = np.complex128)
            The array.
        '''

        ary = self._MSC_to_numpy(self.get_MSC(),
                                 (self.left_subspace.dim, self.right_subspace.dim),
                                 self.left_subspace.idx_to_state,
                                 self.right_subspace.state_to_idx)

        return ary

    ### unary and binary operations

    def __add__(self,x):
        if isinstance(x,Operator):
            return self._op_add(x)
        else:
            raise TypeError('Addition not supported for types %s and %s'
                            % (str(type(self)),str(type(x))))

    def __sub__(self,x):
        return self + -x

    def __neg__(self):
        return -1*self

    def __mul__(self,x):
        if isinstance(x,Operator):
            return self._op_mul(x)
        elif isinstance(x,State):
            return self.get_mat() * x.vec
        else:
            try:
                # check that it is a number, in the
                # most generic way I can think of
                if x == np.array([1]) * x:
                    return self._num_mul(x)
                else:
                    raise ValueError()
            except (TypeError,ValueError):
                raise TypeError('Multiplication not supported for types %s and %s'
                                % (str(type(self)),str(type(x))))

    def __rmul__(self,x):
        if isinstance(x,State):
            return TypeError('Left vector-matrix multiplication not currently '
                             'supported.')
        else:
            return self.__mul__(x)

    def __eq__(self,x):
        if isinstance(x,Operator):
            return np.array_equal(self.get_MSC(),x.get_MSC())
        else:
            raise TypeError('Equality not supported for types %s and %s'
                            % (str(type(self)),str(type(x))))

    def _op_add(self,o):

        if isinstance(o,Sum):
            return Sum(terms = [self] + [o.coeff*t for t in o.terms])
        elif isinstance(o,Operator):
            return Sum(terms = [self,o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

    def _op_mul(self,o):

        if isinstance(o,Product):
            return Product(terms = [self] + [o.coeff*t for t in o.terms])
        elif isinstance(o,Operator):
            return Product(terms = [self,o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

    def _num_mul(self,x):
        o = self.copy()
        o.coeff *= x
        return o

    ### cleanup

    def __del__(self):
        # petsc4py will destroy the matrix object by itself,
        # but for shell matrices the context will not be automatically
        # freed! therefore we need to explicitly call this on object
        # deletion.
        self.destroy_mat()

class Load(Operator):
    """
    Class for operator loaded from memory.
    Only the MSC representation of the operator
    is saved.

    Files should be created with the :meth:`Operator.save`
    method.

    Parameters
    ----------
    fin : str or file object
        The file from which to load the operator.
    """
    def __init__(self,fin):
        self._prop_L = lambda self,value=None: None
        Operator.__init__(self)

        if isinstance(fin,str):
            with open(fin,mode='rb') as f:
                self._load(f)
            self.filename = fin
        else:
            self._load(fin)
            self.filename = None

        self._prop_L = self._postinit_prop_L

    def _load(self,f):
        # figure out the datatype for int
        # TODO: should save data type along with msc representation
        int_t = backend.MSC_dtype[0].newbyteorder('>')
        complex_t = backend.MSC_dtype[2].newbyteorder('>')
        int_size = int_t.itemsize

        self.L = int(np.fromstring(f.read(int_size),dtype=int_t))
        msc_size = int(np.fromstring(f.read(int_size),dtype=int_t))

        self._MSC = np.ndarray(msc_size,dtype=backend.MSC_dtype)

        self._MSC['masks'] = np.fromstring(f.read(int_size*msc_size),dtype=int_t)
        self._MSC['signs'] = np.fromstring(f.read(int_size*msc_size),dtype=int_t)
        self._MSC['coeffs'] = np.fromstring(f.read(complex_t.itemsize*msc_size),dtype=complex_t)

    def _postinit_prop_L(self,value):
        raise TypeError('Cannot set L for operator loaded from file. Value from file: L=%d'%self.L)

    def _build_tex(self,signs='-',request_parens=False):
        t = coeff_to_str(self.coeff,signs=signs)
        if t:
            t += '*'

        if self.filename is None:
            t += r'[\text{operator from file}]'
        else:
            t += r'[\text{operator from file "%s"}]' % self.filename

        return t

    def _get_MSC(self):
        if self.coeff == 1:
            return self._MSC
        else:
            msc = self._MSC.copy()
            msc['coeffs'] *= self.coeff
            return msc

    # TODO: there is no good reason to restrict this behavior. just a couple things to
    # implement
    def _op_add(self,o):
        raise TypeError('Cannot currently use operators loaded from file in expressions.')

    def _op_mul(self,o):
        raise TypeError('Cannot currently use operators loaded from file in expressions.')


class _Expression(Operator):
    def __init__(self,terms,copy=True,L=None):
        """
        Base class for Sum and Product classes.
        """

        Operator.__init__(self,L=L)
        L = self.L

        if copy:
            self.terms = [t.copy() for t in terms]
        else:
            self.terms = list(terms)

        if len(self.terms) == 0:
            raise ValueError('Term list is empty.')

        terms_L = None
        for t in self.terms:
            if t.L is not None:
                if terms_L is not None:
                    if t.L != terms_L:
                        raise ValueError('All terms must have same length L.')
                else:
                    terms_L = t.L

        if len(self.terms) > 1:
            self.min_length = max(o.min_length for o in self.terms)
        elif len(self.terms) == 1:
            self.min_length = self.terms[0].min_length

        # pick up length from terms if it isn't set any other way
        if self.L is None:
            L = terms_L

        self.L = L

    def _get_index_set(self):
        indices = set()
        for term in self.terms:
            indices = indices | term._get_index_set()
        return indices

    def _replace_index(self,ind,rep):
        for term in self.terms:
            term._replace_index(ind,rep)

    def _build_tex(self,signs='-',request_parens=False):
        raise NotImplementedError()

    def _prop_L(self,L):

        # when we are not fully initialized
        if not hasattr(self,'terms'):
            return

        for term in self.terms:
            term.L = L

class Sum(_Expression):
    """
    A sum of several operators. This object can be used in a couple ways.
    All of the following return the exact same object,
    :math:`\sigma^x_0 + \sigma^y_0`\:

    .. code:: python

        Sigmax() + Sigmay()
        Sum([Sigmax(),Sigmay()])
        Sum(s() for s in [Sigmax,Sigmay])

    Parameters
    ----------
    terms : list
        A list of operators to sum

    copy : bool, optional
        Whether to use copies of the operators
        in ``terms`` list, or just use them as-is. Not making
        copies could lead to odd behavior if those operators are
        later modified.

    L : int, optional
        The length of the spin chain
    """

    def __init__(self,terms,copy=True,L=None):

        _Expression.__init__(self,terms,copy,L=L)

        if len(self.terms) > 1:
            self.needs_parens = True

    def _get_MSC(self,shift_index=0,wrap=False):

        all_terms = np.hstack([t.get_MSC(shift_index=shift_index,wrap=wrap) for t in self.terms])
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)

    def _build_tex(self,signs='-',request_parens=False):
        t = coeff_to_str(self.coeff,signs)
        add_parens = False
        if (t and self.needs_parens) or (request_parens and self.needs_parens):
            add_parens = True
            t += r'\left('
        for n,term in enumerate(self.terms):
            t += term.build_tex(signs=('+-' if n else signs))
        if add_parens:
            t += r'\right)'
        return t

    def _op_add(self,o):
        if isinstance(o,Sum):
            return Sum(terms = [self.coeff*t for t in self.terms] + \
                               [o.coeff*t for t in o.terms])
        elif isinstance(o,Operator):
            return Sum(terms = [self.coeff*t for t in self.terms] + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class Product(_Expression):
    """
    A product of several operators. Called in same way as :class:`Sum`.
    For example:

    .. code:: python

        Sigmax() * Sigmay() == Product([Sigmax(),Sigmay()])

    Parameters
    ----------
    terms : list
        A list of operators to multiply

    copy : bool, optional
        Whether to use copies of the operators
        in ``terms`` list, or just use them as-is. Not making
        copies could lead to odd behavior if those operators are
        later modified.

    L : int, optional
        The length of the spin chain
    """

    def __init__(self,terms,copy=True,L=None):

        _Expression.__init__(self,terms,copy,L=L)

        for term in self.terms:
            self.coeff = self.coeff * term.coeff
            term.coeff = 1

    def _get_MSC(self,shift_index=0,wrap=False):

        all_terms = MSC_matrix_product(t.get_MSC(shift_index=shift_index,wrap=wrap) for t in self.terms)
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)

    def _build_tex(self,signs='-',request_parens=False):
        t = coeff_to_str(self.coeff,signs)
        for term in self.terms:
            t += term.build_tex(request_parens=True)
        return t

    def _op_mul(self,o):
        if isinstance(o,Product):
            return Product(terms = [self.coeff*t for t in self.terms] + \
                                   [o.coeff*t for t in o.terms])
        elif isinstance(o,Operator):
            return Product(terms = [self.coeff*t for t in self.terms] + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class _IndexType(Operator):

    def __init__(self,op,min_i=0,max_i=None,index_label='i',wrap=False,L=None):

        Operator.__init__(self,L=L)
        L = self.L

        self._min_i = min_i
        self._max_i = max_i
        self.hard_max = max_i is not None
        self.wrap = wrap

        if self._max_i is not None and self._max_i < self._min_i:
            raise ValueError('max_i must be >= min_i.')

        self.min_length = op.min_length
        if self.L is not None:
            if self._max_i is not None and self._max_i >= self.L:
                raise ValueError('max_i must be < L.')

        if self._min_i < 0:
            raise ValueError('min_i must be >= 0.')

        self.op = op.copy()

        if self.op.L is not None and self.L is None:
            self._L = self.op.L
        self.L = self._L # propagate L

        if not isinstance(index_label,str):
            raise TypeError('Index label should be a string.')
        self.index_label = index_label

        indices = self.op._get_index_set()
        if any(not isinstance(x,int) for x in indices):
            raise TypeError('Can only sum/product over objects with integer indices')
        if min(indices) != 0:
            raise IndexError('Minimum index of summand must be 0.')

        for ind in indices:
            if isinstance(ind,int):
                rep = index_label
                if ind:
                    rep = rep+'+'+str(ind)
                if self.wrap:
                    if ind:
                        rep = '(' + rep + ')'
                    rep = '%L'  # TODO: change L to an integer when it's set
                self.op._replace_index(ind,rep)

        # finally we should set L again to make sure it's set everywhere
        self.L = self.L

    def _get_sigma_tex(self):
        raise NotImplementedError()

    def _build_tex(self,signs='-',request_parens=False):
        t = ''
        if request_parens:
            t += r'\left['
        t += coeff_to_str(self.coeff,signs)
        t += self._get_sigma_tex()+r'_{'+self.index_label+'='+str(self._min_i)+'}'
        if self._max_i is not None:
            t += '^{'+str(self._max_i)+'}'
        else:
            if self.wrap:
                t += '^{L-1}'
            else:
                t += '^{L'+('-'+str(self.min_length))+'}'
        t += self.op.build_tex(request_parens=True)
        if request_parens:
            t += r'\right]'
        return t

    def _get_index_set(self):
        return self.op._get_index_set()

    def _replace_index(self,ind,rep):
        return self.op._replace_index(ind,rep)

    def _prop_L(self,L):

        # not fully initialized yet
        if not hasattr(self,'op'):
            return

        if self.hard_max and L is not None:
            if self._max_i > L-1:
                raise ValueError('Cannot set L smaller than '
                                 'max_i+1 of index sum or product.')
            elif not self.wrap and self._max_i + self.min_length > L:
                raise ValueError('Index sum or product operator would extend '
                                 'past end of spin chain (L too small).')

        self.op.L = L
        if not self.hard_max:
            if L is not None:
                if self.wrap:
                    self._max_i = L-1
                else:
                    self._max_i = L - self.min_length
            else:
                self._max_i = None

    def _get_MSC(self):
        raise NotImplementedError()

class IndexSum(_IndexType):
    """
    The sum of the operator ``op`` translated to sites ``min_i`` through ``max_i``, inclusive.

    For illustrative examples, see the Examples pages.

    Parameters
    ----------
    op : Operator
        The operator to translate along the spin chain

    min_i : int, optional
        The shift index at which to start the sum

    max_i : int or None, optional
        The shift index at which to end the sum (inclusive). If None, sum goes to end of spin chain.

    index_label : str, optional
        The label to use as the index on the sum in the LaTeX representation.

    wrap : bool, optional
        Whether to wrap around, that is, to use periodic boundary conditions.

    L : int, optional
        The length of the spin chain.
    """

    def __init__(self,op,min_i=0,max_i=None,index_label='i',wrap=False,L=None):

        _IndexType.__init__(self,
                            op=op,
                            min_i=min_i,
                            max_i=max_i,
                            index_label=index_label,
                            wrap=wrap,
                            L=L)

    def _get_sigma_tex(self):
        return r'\sum'

    def _get_MSC(self):
        if self._max_i is None:
            raise Exception('Must set L or max_i before building MSC representation of IndexSum.')

        all_terms = np.hstack([self.op.get_MSC(shift_index=i,wrap=self.wrap)
                               for i in range(self._min_i,self._max_i+1)])
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)


class IndexProduct(_IndexType):

    """
    The product of the operator ``op`` translated to sites ``min_i`` through ``max_i``, inclusive.

    For illustrative examples, see the Examples pages.

    Parameters
    ----------
    op : Operator
        The operator to translate along the spin chain

    min_i : int, optional
        The shift index at which to start the product

    max_i : int or None, optional
        The shift index at which to end the product (inclusive).
        If None, product goes to end of spin chain.

    index_label : str, optional
        The label to use as the index on the product in the LaTeX representation.

    wrap : bool, optional
        Whether to wrap around, that is, to use periodic boundary conditions.

    L : int, optional
        The length of the spin chain.
    """

    def __init__(self,op,min_i=0,max_i=None,index_label='i',wrap=False,L=None):

        _IndexType.__init__(self,
                            op=op,
                            min_i=min_i,
                            max_i=max_i,
                            index_label=index_label,
                            wrap=wrap,
                            L=L)

    def _get_sigma_tex(self):
        return r'\prod'

    def _get_MSC(self):
        if self._max_i is None:
            raise Exception('Must set L or max_i before building MSC representation of '
                            'IndexProduct.')
        terms = (self.op.get_MSC(shift_index=i,wrap=self.wrap)
                 for i in range(self._min_i,self._max_i+1))
        all_terms = MSC_matrix_product(terms)
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)


# the bottom level. a single operator (e.g. sigmax)
class _Fundamental(Operator):

    def __init__(self,index=0,L=None):

        Operator.__init__(self,L=L)
        self.index = index
        self.min_length = index + 1
        self.tex = []
        self.tex_end = ''

    def _calc_ind(self,shift_index,wrap):
        ind = self.index+shift_index
        if self.L is not None and ind >= self.L:
            if wrap:
                ind = ind % self.L
            else:
                raise IndexError('requested too large an index')
        return ind

    def _get_index_set(self):
        indices = set()
        for t in self.tex:
            indices = indices | {t[1]}
        return indices

    def _replace_index(self,ind,rep):
        for t in self.tex:
            if t[1] == ind:
                t[1] = rep

    def _build_tex(self,signs='-',request_parens=False):
        t = coeff_to_str(self.coeff,signs)
        for substring,index in self.tex:
            t += substring + str(index)
        t += self.tex_end
        return t

    def _prop_L(self,L):
        pass

    def _get_MSC(self,shift_index=0,wrap=False):
        raise NotImplementedError()

class Sigmax(_Fundamental):
    """
    The Pauli :math:`\sigma_x` operator on site :math:`i`.
    """

    def __init__(self,index=0,L=None):

        if L is None:
            L = config.L

        _Fundamental.__init__(self,index,L=L)
        self.tex = [[r'\sigma^x_{',self.index]]
        self.tex_end = r'}'

    def _get_MSC(self,shift_index=0,wrap=False):
        ind = self._calc_ind(shift_index,wrap)
        return np.array([(1<<ind,0,self.coeff)],dtype=backend.MSC_dtype)


class Sigmaz(_Fundamental):
    """
    The Pauli :math:`\sigma_z` operator on site :math:`i`.
    """

    def __init__(self,index=0,L=None):

        if L is None:
            L = config.L

        _Fundamental.__init__(self,index,L=L)
        self.tex = [[r'\sigma^z_{',self.index]]
        self.tex_end = r'}'

    def _get_MSC(self,shift_index=0,wrap=False):
        ind = self._calc_ind(shift_index,wrap)
        return np.array([(0,1<<ind,self.coeff)],dtype=backend.MSC_dtype)


class Sigmay(_Fundamental):
    """
    The Pauli :math:`\sigma_y` operator on site :math:`i`.
    """

    def __init__(self,index=0,L=None):

        if L is None:
            L = config.L

        _Fundamental.__init__(self,index,L=L)
        self.tex = [[r'\sigma^y_{',self.index]]
        self.tex_end = r'}'

    def _get_MSC(self,shift_index=0,wrap=False):
        ind = self._calc_ind(shift_index,wrap)
        return np.array([(1<<ind,1<<ind,1j*self.coeff)],dtype=backend.MSC_dtype)


class Identity(_Fundamental):
    """
    The identity operator. Since it is tensored with identities on all
    the rest of the sites, the ``index`` argument has no effect.
    """

    def __init__(self,index=0,L=None):

        if L is None:
            L = config.L

        _Fundamental.__init__(self,index,L=L)
        self.tex = []
        self.tex_end = r'I'
        self.min_length = 0

    def _get_MSC(self,shift_index=0,wrap=False):
        return np.array([(0,0,self.coeff)],dtype=backend.MSC_dtype)


# also should hide this tex when appropriate
class Zero(_Fundamental):
    """
    The zero operator---equivalent to a matrix of all zeros of dimension :math:`2^L`.
    Like for the identity, the ``index`` argument has no effect.
    """

    def __init__(self,index=0,L=None):

        if L is None:
            L = config.L

        _Fundamental.__init__(self,index,L=L)
        self.tex = []
        self.tex_end = r'0'
        self.min_length = 0

    def _get_MSC(self,shift_index=0,wrap=False):
        return np.array([],dtype=backend.MSC_dtype)
