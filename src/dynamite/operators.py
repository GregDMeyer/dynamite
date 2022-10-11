"""
This module provides the building blocks for Hamiltonians, and
defines their built-in behavior and operations.
"""

from itertools import chain
from zlib import crc32
import re
from string import digits, ascii_lowercase
import numpy as np

from . import config, validate, msc_tools
from .computations import evolve, eigsolve
from .subspaces import Full, Explicit
from .states import State
from .tools import complex_enabled

class Operator:
    """
    A class representing a quantum operator.

    This class generally won't be directly instantiated by the user, but is returned by the
    other functions in this module.
    """

    def __init__(self):
        self._max_spin_idx = None
        self._mats = {}
        self._msc = None
        self._is_reduced = False
        self._shell = config.shell
        self._allow_projection = False

        if config.subspace is not None:
            self._subspaces = [(config.subspace, config.subspace)]
        else:
            self._subspaces = [(Full(), Full())]

        if config.L is not None:
            self.L = config.L

        self._string_rep = _OperatorStringRep()

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

        rtn._string_rep = self._string_rep.copy()

        return rtn

    ### computations

    def evolve(self, state, t, **kwargs):
        r"""
        Time-evolve a state, using the operator as the Hamiltonian.

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
            if self.msc is None:
                self._max_spin_idx = 0
            else:
                self._max_spin_idx = msc_tools.max_spin_idx(self.msc)

        return self._max_spin_idx

    @property
    def L(self):
        """
        Property representing the length of the spin chain.
        """
        self._update_L_from_subspaces()
        return self.left_subspace.L

    def _update_L_from_subspaces(self):
        '''
        Propagate L to all subspaces if it has been set in any
        one of them.
        '''
        L = None
        for subspaces in self._subspaces:
            for subspace in subspaces:
                if subspace.L is not None:
                    if L is None:
                        L = subspace.L
                    elif L != subspace.L:
                        raise ValueError('All subspaces of an operator must '
                                         'have the same spin chain length L.')

        # propagate the value to all subspaces
        if L is not None:
            self.L = L

    @L.setter
    def L(self, value):
        L = validate.L(value)
        if L < self.max_spin_idx + 1:
            raise ValueError('Cannot set L smaller than one plus the largest spin index'
                             'on which the operator has support (max_spin_idx = %d)' %
                             (self.max_spin_idx))
        for left, right in self._subspaces:
            left.L = L
            right.L = L

    def establish_L(self):
        '''
        If L has not been set, set it to the minimum value possible based on the
        support of the operator (and propagate that value to all registered
        subspaces). Does nothing if the operator already has a value for L
        other than `None`.
        '''
        self.L = self.get_length()

    def get_length(self):
        '''
        Returns the length of the spin chain for this operator. It is defined by the
        property :meth:`Operator.L` if it has been set by the user. Otherwise, one
        plus the largest spin index on which the operator has support is returned.
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
        self.establish_L()
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

    def infinity_norm(self, subspaces=None):
        """
        Computes the infinity norm of the operator's matrix, on the given
        subspace(s). If subspace is not supplied, defaults to the operator's
        default subspace (the most recently added one, or Full).

        Parameters
        ----------
        subspaces : tuple(Subspace, Subspace), optional
            The subspace pair on which to compute the infinity norm

        Returns
        -------
        float
            The norm
        """
        config._initialize()
        from petsc4py import PETSc
        return self.get_mat(subspaces=subspaces).norm(PETSc.NormType.INFINITY)

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
        return self.get_subspace_list()[-1][0]

    @property
    def right_subspace(self):
        """
        Get the default right subspace for this operator. This is the subspace most recently
        added with :meth:`Operator.add_subspace`, or config.subspace if
        :meth:`Operator.add_subspace` has not been called.
        """
        return self.get_subspace_list()[-1][1]

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
        else:
            if (left is not right and
                (not left.product_state_basis or not right.product_state_basis)):
                raise ValueError("subspaces must be the same object if either is not a "
                                 "product state basis")

        left = validate.subspace(left)
        right = validate.subspace(right)

        # L should become the subspace spin chain length if it's not set
        if self.L is None:
            if left.L is not None:
                self.L = left.L
            elif right.L is not None:
                self.L = right.L

        # now if the operator's L is set, both subspaces' L should equal it
        if self.L is not None:
            for subspace in (left, right):
                if subspace.L is None:
                    subspace.L = self.L
                elif subspace.L != self.L:
                    raise ValueError('operator and subspaces must all have '
                                     'same spin chain length')

        if not self.has_subspace(left, right):
            self.get_subspace_list().append((left, right))

    def get_subspace_list(self):
        '''
        Return a list of the subspaces that have been registered for this operator.
        '''
        self._update_L_from_subspaces()
        return self._subspaces

    def has_subspace(self, left, right=None):
        '''
        Check if a subspace or pair of subspaces has been added to the
        operator.

        Parameters
        ----------

        left : dynamite.subspaces.Subspace
            The left subspace

        right : dynamite.subspaces.Subspace, optional
            The right subspace. If omitted,
            the left subspace is reused for the right.
        '''
        if right is None:
            right = left

        for (left_s, right_s) in self.get_subspace_list():
            if left.identical(left_s) and right.identical(right_s):
                return True

        return False

    def conserves(self, left, right=None):
        """
        Return whether the operator conserves the given subspace. If both
        ``left`` and ``right`` are supplied, return whether the image of the
        operator when applied to the ``right`` subspace is completely
        contained in the ``left`` subspace.
        """
        self.establish_L()

        if right is None:
            right = left

        if not left.product_state_basis or not right.product_state_basis:
            if left is not right:
                raise ValueError('if left or right subspace is not a product '
                                 'state basis, they must be identical')

        left.L = self.L
        right.L = self.L

        self.reduce_msc()
        if not left.product_state_basis:
            msc, conserved = left.reduce_msc(self.msc, check_conserves=True)
            if not conserved:
                return False
        else:
            msc = self.msc

        masks, mask_offsets = self._get_mask_offsets(msc)

        config._initialize()
        from ._backend import bpetsc

        return bpetsc.check_conserves(
            masks=masks,
            mask_offsets=mask_offsets,
            signs=np.ascontiguousarray(msc['signs']),
            coeffs=np.ascontiguousarray(msc['coeffs']),
            left_type=left.to_enum(),
            left_data=left.get_cdata(),
            right_type=right.to_enum(),
            right_data=right.get_cdata(),
        )

    @property
    def allow_projection(self):
        """
        Whether to allow subspaces for which matrix multiplication implements
        a projection (those for which ``Operator.conserves(subspace)`` or
        ``Operator.conserves(left_subspace, right_subspace)`` returns False).
        """
        return self._allow_projection

    @allow_projection.setter
    def allow_projection(self, value):
        self._allow_projection = value

    ### text representations

    def __str__(self):
        return self._string_rep.string

    def __repr__(self):
        rtn = f'<Operator on {self.get_length()} spins: '
        rtn += str(self)
        rtn += '>'
        return rtn

    # for jupyter notebooks
    def _repr_latex_(self):
        return '$' + self._string_rep.get_latex() + '$'

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

    ### save to disk

    def serialize(self):
        '''
        Serialize the operator into a string of bytes.
        The byte string ONLY contains dynamite's internal representation of
        the operator. It does not include any other information, such as
        subspaces etc.

        Returns
        -------
        bytes
            The byte string containing the serialized object.

        '''
        return msc_tools.serialize(self.msc)

    @classmethod
    def from_bytes(cls, data):
        """
        Load operator from a byte string generated with the
        :meth:`Operator.serialize` method.

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
        o._string_rep.string = '[operator from bytes]'
        o._string_rep.tex = r'\left[\text{operator from bytes}\right]'
        return o

    def save(self, filename):
        """
        Save the operator to disk. Can be loaded again via the
        :meth:`Operator.load` method. Only saves the operator itself, not any
        associated subspaces or other data.

        Parameters
        ----------
        filename : str
            The path to the file to save the operator in.
        """
        config._initialize()
        from petsc4py import PETSc
        if PETSc.COMM_WORLD.rank == 0:
            with open(filename, mode='wb') as f:
                f.write(self.serialize())

        PETSc.COMM_WORLD.barrier()

    @classmethod
    def load(cls, filename):
        '''
        Load the operator in file ``filename`` and return the corresponding
        object.

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
            op = cls.from_bytes(bytestring)
        return op

    ### interface with PETSc

    def get_mat(self, subspaces=None):
        """
        Get the PETSc matrix corresponding to this operator, building it if necessary.

        Parameters
        ----------
        subspaces : tuple(Subspace, Subspace), optional
            The subspace pair to get the matrix for. If the matrix is already built for this
            pair, it will be reused. If this option is omitted, the last subspace added with
            :meth:`Operator.add_subspace` will be used, or the Full space by default.

        Returns
        -------
        petsc4py.PETSc.Mat
            The PETSc matrix corresponding to the operator.
        """
        if subspaces is None:
            subspaces = (self.left_subspace, self.right_subspace)

        if subspaces not in self._mats:
            self.build_mat(subspaces)

        return self._mats[subspaces]

    def build_mat(self, subspaces=None):
        """
        Build the PETSc matrix, destroying any matrix that has already been built, and
        store it internally. This function does not return the matrix--see
        :meth:`Operator.get_mat` for that functionality. This function is rarely needed
        by the end user, since it is called automatically whenever the underlying matrix
        needs to be built or rebuilt.
        """
        if subspaces is None:
            subspaces = (self.left_subspace, self.right_subspace)

        if not self.has_subspace(*subspaces):
            raise ValueError('Attempted to build matrix for a subspace that has not '
                             'been added to the operator.')

        config._initialize()
        from ._backend import bpetsc

        self.reduce_msc()

        if not subspaces[0].product_state_basis:
            msc = self.subspace.reduce_msc(self.msc)
        else:
            msc = self.msc

        self._check_consistent_msc(msc)

        if not self.allow_projection and not self.conserves(*subspaces):
            raise ValueError("Constructing the operator's matrix on this "
                             "subspace yields a projection (e.g. subspace is "
                             "not conserved by the operator). If this "
                             "behavior is desired, set the "
                             "Operator.allow_projection parameter to True.")

        if not msc_tools.is_hermitian(msc):
            raise ValueError('Building non-Hermitian matrices currently not supported.')

        masks, mask_offsets = self._get_mask_offsets(msc)

        mat = bpetsc.build_mat(
            masks=np.ascontiguousarray(masks),
            mask_offsets=np.ascontiguousarray(mask_offsets),
            signs=np.ascontiguousarray(msc['signs']),
            coeffs=np.ascontiguousarray(msc['coeffs']),
            left_type=subspaces[0].to_enum(),
            left_data=subspaces[0].get_cdata(),
            right_type=subspaces[1].to_enum(),
            right_data=subspaces[1].get_cdata(),
            shell=self.shell,
            gpu=config.gpu
        )

        self._mats[subspaces] = mat

    @classmethod
    def _check_consistent_msc(cls, msc):
        config._initialize()
        from petsc4py import PETSc

        # msc cannot be inconsistent with only 1 rank
        if PETSc.COMM_WORLD.size == 1:
            return

        comm = PETSc.COMM_WORLD.tompi4py()

        checksum = crc32(msc.data)
        all_checksums = comm.allgather(checksum)

        if not all(v == all_checksums[0] for v in all_checksums):
            msg = "operator is inconsistent across MPI ranks. Was it " + \
                "constructed using non-deterministic code, such as random " + \
                "numbers with inconsistent seeds?"
            raise RuntimeError(msg)

    @classmethod
    def _get_mask_offsets(cls, msc):
        """
        Return an array of unique mask values, and the indices where each starts
        """
        if not np.all(np.diff(msc['masks']) >= 0):
            raise ValueError('msc must be sorted first')

        masks, indices = np.unique(msc['masks'], return_index=True)

        # need to add the last index
        mask_offsets = np.ndarray((indices.size+1,),
                                  dtype=msc.dtype['masks'])
        mask_offsets[:-1] = indices
        mask_offsets[-1] = msc.shape[0]

        return masks, mask_offsets

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

    def estimate_memory(self, mpi_size=None):
        '''
        Estimate the total amount of memory that will be used by this
        operator once the matrix is constructed (for example, after
        calling ``.evolve()``), summed across all MPI ranks.

        Note that the memory allocated for communication etc. depends on
        a number of different parameters, so actual memory usage may vary.

        For operators containing terms that cancel to zero in some cases
        (such as XX+YY), this function will overestimate the required
        memory for non-shell matrices.

        Parameters
        ----------

        mpi_size : int, optional
            The number of ranks to estimate memory usage for. If not
            provided, the mpi size of the running program is used.

        Returns
        -------
        float
            The expected memory usage, in gigabytes
        '''
        if mpi_size is None:
            config._initialize()
            from petsc4py import PETSc
            mpi_size = PETSc.COMM_WORLD.size

        if self.shell:
            usage_bytes = self.msc.nbytes

            # Explicit is the only subspace that uses an appreciable
            # amount of memory
            for sp in (self.left_subspace, self.right_subspace):
                if isinstance(sp, Explicit):
                    usage_bytes += sp.state_map.nbytes
                    usage_bytes += sp.rmap_indices.nbytes
                    usage_bytes += sp.rmap_states.nbytes

            # these values are stored redundantly on every rank
            usage_bytes *= mpi_size

        else:
            int_size = msc_tools.dnm_int_t().itemsize
            scalar_size = 16 if complex_enabled() else 8
            elem_size = int_size + scalar_size

            # because we have to add a zero diagonal if it doesn't exist
            # to keep PETSc happy
            nnz = self.nnz
            if np.all(self.msc['masks']):
                nnz += 1

            usage_bytes = nnz*self.dim[0]*elem_size

            # the pointers to the array for each row and
            # other bookkeeping arrays petsc allocates
            usage_bytes += 5*int_size*self.dim[0]

            if mpi_size > 1:
                usage_bytes += 3*int_size*self.dim[0]  # indices for sparse mat
                usage_bytes += scalar_size*self.dim[1] # VecCreate_Seq
                usage_bytes += int_size*self.dim[0]    # MatSetUpMultiply
                usage_bytes += 2*int_size*self.dim[1]  # VecScatterCreate
                usage_bytes += 2*int_size*self.dim[0]  # MatMarkDiagonal

        return usage_bytes/1E9

    def create_states(self):
        '''
        Return a bra and ket compatible with this matrix.

        Returns
        -------
        tuple
            The two states
        '''
        self.establish_L()

        bra = State(subspace=self.left_subspace)
        ket = State(subspace=self.right_subspace)
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

    def truncate(self, tol=1e-12):
        '''
        Remove terms whose magnitude (absolute value) is less than `tol`.

        Parameters
        ----------
        tol : float
            The cutoff for truncation
        '''
        self.msc = msc_tools.truncate(self.msc, tol=tol)

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
        self.establish_L()

        if subspaces is None:
            subspaces = (self.left_subspace, self.right_subspace)

        self.reduce_msc()

        if not subspaces[0].product_state_basis:
            msc = self.subspace.reduce_msc(self.msc)
        else:
            msc = self.msc

        ary = msc_tools.msc_to_numpy(msc,
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
            if x == 0:
                return self.copy()
            else:
                x = x*identity()
        return self._op_add(x)

    def __radd__(self, x):
        if not isinstance(x, Operator):
            if x == 0:
                return self.copy()
            else:
                x = x*identity()
        return x + self

    def __sub__(self, x):
        return self + -x

    def __rsub__(self, x):
        return x + -self

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

    def __truediv__(self, x):
        if isinstance(x, Operator):
            raise TypeError('Dividing by Operators not supported.')

        return (1/x) * self

    def __eq__(self, x):
        if isinstance(x, Operator):
            self.reduce_msc()
            x.reduce_msc()
            return np.array_equal(self.msc, x.msc)
        else:
            raise TypeError('Equality not supported for types %s and %s'
                            % (str(type(self)), str(type(x))))

    def _op_add(self, o):
        self._check_compatible(o)

        rtn = self.copy()
        rtn.msc = msc_tools.msc_sum([self.msc, o.msc])
        rtn._string_rep.string = str(self) + ' + ' + str(o)
        rtn._string_rep.tex = self._string_rep.tex + ' + ' + o._string_rep.tex
        rtn._string_rep.brackets = '()'
        return rtn

    def _op_mul(self, o):
        self._check_compatible(o)

        rtn = self.copy()
        rtn.msc = msc_tools.msc_product([self.msc, o.msc])

        rtn._string_rep.string = self._string_rep.with_brackets('string') + '*'
        rtn._string_rep.string += o._string_rep.with_brackets('string')

        rtn._string_rep.tex = self._string_rep.with_brackets('tex')
        rtn._string_rep.tex += o._string_rep.with_brackets('tex')

        rtn._string_rep.brackets = ''

        return rtn

    def _check_compatible(self, o):
        """
        Check that two operators are compatible to be combined (added or
        multiplied).
        """
        if self.shell != o.shell:
            raise ValueError("Operators must have the same value of the "
                             "'shell' parameter to be combined. To set it "
                             "globally, set dynamite.config.shell")

        if self.allow_projection != o.allow_projection:
            raise ValueError("Operators must have the same value of the "
                             "'allow_projection' parameter to be combined.")

        if self.L != o.L:
            raise ValueError("Operators to be combined must have the same "
                             "value of the spin chain length L. To set it "
                             "globally, set dynamite.config.L")

        subsp_1 = self.get_subspace_list()
        subsp_2 = o.get_subspace_list()

        subspaces_bad = False
        if len(subsp_1) != len(subsp_2):
            subspaces_bad = True
        else:
            for (i, (left_1, right_1)) in enumerate(subsp_1):
                # for efficiency, start later
                for (left_2, right_2) in chain(subsp_2[i:], subsp_2[:i]):
                    if left_1.identical(left_2) and right_1.identical(right_2):
                        break
                else:
                    subspaces_bad = True
                    break

        if subspaces_bad:
            raise ValueError("Operators to be combined must have the same "
                             "subspaces. To set a global default subspace, "
                             "set dynamite.config.subspace")

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
        x.assert_initialized()

        self.establish_L()

        right_subspace = x.subspace
        right_match = [(left, right) for left, right in self.get_subspace_list()
                       if right.identical(right_subspace)]
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
        result.set_initialized()
        return result

    def _vec_mul(self, x):
        return self.dot(x)

    def scale(self, x):
        '''
        Scale an operator by a numerical value without making a copy. This is
        more efficient than just doing x*Operator.

        Parameters
        ----------
        x : numeric type
            The coefficient to scale by
        '''
        try:
            self.msc['coeffs'] *= x
        except (ValueError, TypeError):
            raise TypeError(f'Cannot scale operator by type {type(x)}')

        # coefficient up to 3 digits of precision, with trailing zeros removed
        coeff_str = f'{x:.3f}'.rstrip('0').rstrip('.')

        self._string_rep.string = coeff_str + self._string_rep.with_brackets('string')
        self._string_rep.tex = coeff_str + self._string_rep.with_brackets('tex')
        self._string_rep.brackets = ''

    def _num_mul(self, x):
        rtn = self.copy()
        rtn.scale(x)
        return rtn


class _OperatorStringRep:
    '''
    This class builds and manages the string and LaTeX representations of an
    operator, to implement the __str__, __repr__, and _repr_latex methods
    of the Operator class.
    '''

    def __init__(self, string=None, tex=None, brackets=None):
        if string is None:
            string = '[operator]'

        if tex is None:
            tex = r'\[\text{operator}\]'

        if brackets is None:
            brackets = ''

        self._string = string
        self._tex = tex
        self._brackets = brackets

    def copy(self):
        return _OperatorStringRep(self.string, self.tex, self.brackets)

    @property
    def string(self):
        '''
        The text string that will be returned when ``str(obj)`` is called.
        '''
        return self._string

    @string.setter
    def string(self, value):
        self._string = value

    @property
    def tex(self):
        '''
        A LaTeX expression corresponding to the object. Can be set to any
        valid TeX math expression.
        '''
        return self._tex

    @tex.setter
    def tex(self, value):
        self._tex = value

    @property
    def brackets(self):
        '''
        Which kind of brackets to surround the expression with. Options are
        ``'()'``, ``'[]'``, or ``''``, where the empty string means no
        brackets.
        '''
        return self._brackets

    @brackets.setter
    def brackets(self, value):
        if value not in ['()', '[]', '']:
            raise ValueError("Brackets must be one of '()', '[]', or ''")
        self._brackets = value

    def with_brackets(self, which):
        '''
        Return a string or tex representation of the object, surrounded by
        brackets if necessary. Useful for building larger expressions.

        Parameters
        ----------

        which : str
            Whether to return a normal string or tex. Options are ``'string'``
            or ``'tex'``.
        '''
        if which == 'tex':
            base = self.tex
            brackets = [
                x+y for x, y in zip([r'\left', r'\right'], self.brackets)
            ]
        elif which == 'string':
            base = self.string
            brackets = self.brackets
        else:
            raise ValueError("which must be either 'string' or 'tex'.")

        if not self.brackets:
            return base

        return base.join(brackets)

    def __repr__(self):
        rtn = f"_OperatorStringRep('{self.string}', '{self.tex}', "
        rtn += f"'{self.brackets}')"
        return rtn

    def get_latex(self):
        '''
        Return a clean LaTeX representation (with all replacements performed).
        '''
        return self.tex.replace('{IDX', '{')


def load_from_file(filename):
    '''
    DEPRECATED: use dynamite.operators.Operator.load
    '''
    raise DeprecationWarning("operators.load_from_file is deprecated; "
                             "use operators.Operator.load")


def from_bytes(data):
    """
    DEPRECATED: use dynamite.operators.Operator.from_bytes
    """
    raise DeprecationWarning("operators.from_bytes is deprecated; "
                             "use operators.Operator.from_bytes")


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
        strings.append(t._string_rep.string)
        texs.append(t._string_rep.tex)
        if n >= nshow:
            break
    else:
        done = True

    if not done:
        strings[-1] = '...'
        texs[-1] = r'\cdots'
        msc_terms.append(msc_tools.msc_sum(t.msc for t in iterterms))

    o.msc = msc_tools.msc_sum(msc_terms)
    o._string_rep.string = ' + '.join(strings)
    o._string_rep.tex = ' + '.join(texs)
    o._string_rep.brackets = '()'
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
        strings.append(t._string_rep.with_brackets('string'))
        texs.append(t._string_rep.with_brackets('tex'))

    if msc_terms:
        o = Operator()
        o.msc = msc_tools.msc_product(msc_terms)
        o._string_rep.string = '*'.join(strings)
        o._string_rep.tex = ''.join(texs)
        o._string_rep.brackets = ''
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

    size = validate.L(size)

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

    rtn._string_rep.string = 'index_sum(' + str(op) + ', sites %d - %d' % (start, stop-1)
    if boundary == 'closed':
        rtn._string_rep.string += ', wrapped)'
    else:
        rtn._string_rep.string += ')'

    # add i to the indices for TeX representation
    sub_tex = op._string_rep.with_brackets('tex')
    idx = _get_next_index(sub_tex)
    sub_tex = sub_tex.replace('{IDX', '{IDX'+idx+'+')
    sub_tex = sub_tex.replace('{IDX'+idx+'+0', '{IDX'+idx)

    rtn._string_rep.tex = r'\sum_{'+idx+'=%d}^{%d}' % (start, stop-1) + sub_tex
    rtn._string_rep.brackets = '[]'

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

    size = validate.L(size)

    stop = start + size - op.max_spin_idx

    rtn = Operator()
    rtn.msc = msc_tools.msc_product(op.get_shifted_msc(i, wrap_idx = None) for i in range(start, stop))

    rtn._string_rep.string = 'index_product(' + str(op) + ', sites %d - %d)' % (start, stop-1)

    # add i to the indices for TeX representation
    sub_tex = op._string_rep.with_brackets('tex')
    idx = _get_next_index(sub_tex)
    sub_tex = sub_tex.replace('{IDX', '{IDX'+idx+'+')
    sub_tex = sub_tex.replace('{IDX'+idx+'+0', '{IDX'+idx)
    rtn._string_rep.tex = r'\prod_{'+idx+'=%d}^{%d}' % (start, stop-1)
    rtn._string_rep.tex += sub_tex
    rtn._string_rep.brackets = '[]'

    return rtn


def _get_next_index(tex_str):
    if '{IDX' not in tex_str:
        return 'i'

    max_idx = max(tex_str[m.end()] for m in re.finditer('{IDX', tex_str))

    if max_idx in ascii_lowercase:
        return ascii_lowercase[(ascii_lowercase.find(max_idx)+1) % 26]

    else:
        return 'i'


def sigmax(i=0):
    r"""
    The Pauli :math:`\sigma_x` operator on site :math:`i`.
    """
    i = validate.spin_index(i)

    o = Operator()
    o.msc = [(1<<i, 0, 1)]
    o._string_rep.tex = r'\sigma^x_{IDX'+str(i)+'}'
    o._string_rep.string = 'σx'+str(i).join('[]')
    return o

def sigmay(i=0):
    r"""
    The Pauli :math:`\sigma_y` operator on site :math:`i`.
    """
    i = validate.spin_index(i)

    o = Operator()
    o.msc = [(1<<i, 1<<i, 1j)]
    o._string_rep.tex = r'\sigma^y_{IDX'+str(i)+'}'
    o._string_rep.string = 'σy'+str(i).join('[]')
    return o

def sigmaz(i=0):
    r"""
    The Pauli :math:`\sigma_z` operator on site :math:`i`.
    """
    i = validate.spin_index(i)

    o = Operator()
    o.msc = [(0, 1<<i, 1)]
    o._string_rep.tex = r'\sigma^z_{IDX'+str(i)+'}'
    o._string_rep.string = 'σz'+str(i).join('[]')
    return o

def sigma_plus(i=0):
    r"""
    The :math:`\sigma_+ = \sigma_x + i \sigma_y` operator.

    .. note::

        :math:`\sigma_+ = \left( \begin{array}{cc} 0 & 2 \\ 0 & 0 \\ \end{array} \right)`,
        so :math:`S_+ = \left( \begin{array}{cc} 0 & 1 \\ 0 & 0 \\ \end{array} \right) = \frac{1}{2} \sigma_+`
    """
    i = validate.spin_index(i)

    o = sigmax(i) + 1j*sigmay(i)
    o._string_rep.tex = r'\sigma^+_{IDX'+str(i)+'}'
    o._string_rep.string = 'σ+'+str(i).join('[]')
    return o

def sigma_minus(i=0):
    r"""
    The :math:`\sigma_- = \sigma_x - i \sigma_y` operator.

    .. note::

        :math:`\sigma_- = \left( \begin{array}{cc} 0 & 0 \\ 2 & 0 \\ \end{array} \right)`,
        so :math:`S_- = \left( \begin{array}{cc} 0 & 0 \\ 1 & 0 \\ \end{array} \right) = \frac{1}{2} \sigma_-`
    """
    i = validate.spin_index(i)

    o = sigmax(i) - 1j*sigmay(i)
    o._string_rep.tex = r'\sigma^-_{IDX'+str(i)+'}'
    o._string_rep.string = 'σ-'+str(i).join('[]')
    return o

def identity():
    """
    The identity operator.
    """
    o = Operator()
    o.msc = [(0, 0, 1)]
    o._string_rep.tex = '𝟙'
    o._string_rep.string = '1'
    return o

def zero():
    """
    The zero operator---equivalent to a matrix of all zeros.
    """
    o = Operator()
    o.msc = []
    o._string_rep.tex = '0'
    o._string_rep.string = '0'
    return o
