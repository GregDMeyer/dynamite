
from . import config, validate, subspaces
from .tools import complex_enabled
from .msc_tools import dnm_int_t

import numpy as np
from os import urandom
from time import time
import pickle


class State:
    """
    Class representing a state.

    Parameters
    ----------

    L : int, optional
        Spin chain length. Can be ommitted if config.L is set or a subspace
        is provided.

    subspace : dynamite.subspace.Subspace, optional
        The subspace on which the state should be defined.
        If not specified, defaults to config.subspace.

    state : int or str, optional
        An initial product state to set the state to. Also accepts ``'random'``. The
        state can also be initialized later with the :meth:`set_product` and
        :meth:`set_random` methods.

    seed : int, optional
        If the ``state`` argument is set to ``'random'``, the seed for the random number
        generator. This argument is ignored otherwise.
    """

    def __init__(self, L = None, subspace = None, state = None, seed = None):

        if subspace is None:
            if config.subspace is not None:
                subspace = config.subspace
            else:
                subspace = subspaces.Full()

        if L is None and subspace.L is None:
            L = config.L

        if subspace.L is None:
            if L is None:
                raise ValueError('Must specify L either as a parameter, '
                                 'by providing a subspace, or via config.L')
            else:
                subspace.L = validate.L(L)
        elif L is not None and L != subspace.L:
            raise ValueError('The value of L provided as a parameter '
                             'does not match that of the subspace')

        self._subspace = validate.subspace(subspace)
        self._vec = None  # create when first used
        self._initialized = False

        # whether to use binary or letters/arrows for
        # string representation
        self.repr_binary = True

        if state is not None:
            if state == 'random':
                self.set_random(seed=seed)
            else:
                self.set_product(state)

    @property
    def L(self):
        return self.subspace.L

    def copy(self, result=None):
        if result is None:
            result = State(self.L, self.subspace.copy())

        if self.subspace != result.subspace:
            raise ValueError('subspace of state and result must match')

        if self.initialized:
            self.vec.copy(result.vec)
            result.set_initialized()
        elif result.initialized:
            raise UninitializedError('Cannot copy from uninitialized state to '
                                     'one that has been initialized')

        return result

    @property
    def subspace(self):
        """
        The space on which the vector is defined.

        See :mod:`dynamite.subspaces` for details.
        """
        return self._subspace

    @property
    def vec(self):
        """
        The PETSc vector containing the state data.

        petsc4py Vec methods can be used through this interface--for example, to find the norm of a
        State `s`, one can do `state.vec.norm()`. The methods don't seem to be documented anywhere,
        but are fairly transparent from looking at the petsc4py source code.
        """
        if self._vec is None:
            config._initialize()
            from petsc4py import PETSc

            self._vec = PETSc.Vec().create()
            self._vec.setSizes(self.subspace.get_dimension())
            self._vec.setFromOptions()

        return self._vec

    @property
    def initialized(self):
        '''
        Whether the state vector data has been set yet.
        '''
        return self._initialized

    def set_initialized(self):
        '''
        Mark that the state vector data has been set. This method does not
        normally need to be called by the user, unless the vector data is
        set manually by directly accessing the underlying PETSc vector.
        '''
        self._initialized = True

    def assert_initialized(self):
        '''
        Raise an exception if the vector has not been initialized
        '''
        if not self.initialized:
            raise UninitializedError("State vector data has not been set yet")

    @classmethod
    def str_to_state(cls, s, L):
        '''
        Convert a string to an integer whose bitwise representation is the spin
        configuration (0=↑, 1=↓) of a product state. The characters
        'D' and 'U' represent down and up spins, like ``"DUDDU...UDU"`` (D=↓, U=↑).

        .. note::
            For the string representation, the leftmost character is spin index 0. For
            an integer representation, the rightmost (least significant) bit is!

        Parameters
        ----------
        s : int or string
            The state. If an integer is passed, the same integer is returned.

        L : int
            The length of the spin chain

        Returns
        -------
        int
            The state
        '''

        if isinstance(s, str):
            if len(s) != L:
                raise ValueError('state string must have length L')

            if not all(c in ['U', 'D', '0', '1'] for c in str(s)):
                raise ValueError('only character U, D, 0, or 1 allowed in '
                                 'state string')

            state = 0
            for i,c in enumerate(s):
                if c in ('D', '1'):
                    state += 1<<i

        else:
            state = int(s)

            # ensure that it is valid, to the extent possible
            # essentially, it just should not have a bit set at a
            # position above L
            if state >> L != 0:
                raise ValueError(f"value (binary: {bin(s)[2:]}) does not "
                                 "correspond to a valid state of length L")

        return state

    def set_product(self, s):
        """
        Initialize to a product state. Can be specified either be an integer
        whose binary representation represents the spin configuration
        (0=↑, 1=↓) of a product state, or a string of the form
        ``"DUDDU...UDU"`` (D=↓, U=↑) or ``"10110...010"`` (1=↓, 0=↑). If it is
        a string, the string's length must equal ``L``.

        .. note:
            In integer representation, the least significant bit represents
            spin 0. So, if you look at a binary representation of the integer
            (for example with Python's `bin` function) spin 0 will be the
            rightmost bit!

        Parameters
        ----------

        s : int or str
            A representation of the state.
        """

        s2i_result = self.subspace.state_to_idx(self.str_to_state(s, self.L))
        if isinstance(s2i_result, tuple):
            idx, sign = s2i_result
        else:
            idx = s2i_result

        if idx == -1:
            raise ValueError('Provided initial state not in requested subspace.')

        self.vec.set(0)

        istart, iend = self.vec.getOwnershipRange()
        if istart <= idx < iend:
            self.vec[idx] = 1

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

        if isinstance(s, str) and any(c in ('0', '1') for c in s):
            self.repr_binary = True
        else:
            self.repr_binary = False

        self.set_initialized()

    @classmethod
    def generate_time_seed(cls):

        config._initialize()
        from petsc4py import PETSc

        if PETSc.COMM_WORLD.size == 1:
            # in this case it works without needing mpi4py
            return int(time())

        else:
            # otherwise we have to coordinate among the ranks
            CW = PETSc.COMM_WORLD.tompi4py()

            if CW.rank == 0:
                seed = int(time())
            else:
                seed = None

            return CW.bcast(seed, root = 0)

    def set_random(self, seed = None, normalize = True):
        """
        Initialized to a normalized random state.

        .. note::
            When running under MPI with multiple processes, the seed is incremented
            by the MPI rank, so that each process generates different random values.

        Parameters
        ----------

        seed : int, optional
            A seed for numpy's PRNG that is used to build the random state. The user
            should pass the same value on every process.

        normalize : bool
            Whether to rescale the random state to have norm 1.
        """

        config._initialize()
        from petsc4py import PETSc

        istart, iend = self.vec.getOwnershipRange()

        R = np.random.RandomState()

        if seed is None:
            try:
                seed = int.from_bytes(urandom(4), 'big', signed=False)
            except NotImplementedError:
                seed = self.generate_time_seed()

        # if my code is still being used in year 2038, wouldn't want it to
        # overflow numpy's PRNG seed range ;)
        R.seed((seed + PETSc.COMM_WORLD.rank) % 2**32)

        local_size = iend-istart

        if complex_enabled():
            self.vec[istart:iend] =    R.standard_normal(local_size) + \
                                    1j*R.standard_normal(local_size)
        else:
            self.vec[istart:iend] = R.standard_normal(local_size)

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

        if normalize:
            self.vec.normalize()

        self.set_initialized()

    def project(self, index, value):
        '''
        Project the state onto a subspace in which the qubit
        at the given index has been projected onto the given
        value (0 or 1). In other words, perform a projective
        measurement, post-selecting on the outcome.

        Projection done in-place, function returns None.

        Parameters
        ----------
        index : int
            The spin index to project

        value : 0 or 1
            Which single-spin state to project onto
        '''
        self.assert_initialized()

        if index < 0 or index >= self.subspace.L:
            raise ValueError("spin index out of range")

        if value not in (0, 1):
            raise ValueError("value must be 0 or 1")

        # deal with MPI vectors (this process might not own all of it)
        istart, iend = self.vec.getOwnershipRange()
        idxs = np.arange(istart, iend, dtype=dnm_int_t)

        state_list = self.subspace.idx_to_state(idxs)
        idxs_to_zero = idxs[((state_list >> index) & 1) != value]

        # this is sad because you are allocating a whole vector of zeros,
        # but alas, petsc4py requires a vector...
        self.vec[idxs_to_zero] = np.zeros(len(idxs_to_zero))
        self.vec.assemble()

        self.vec.normalize()

    @classmethod
    def _to_numpy(cls, vec, to_all = False):
        '''
        Collect PETSc vector (split across processes) to a
        numpy array on process 0, or to all processes if
        `to_all == True`.

        Parameters
        ----------
        vec : petsc4py.PETSc.Vec
            The input vector

        to_all : bool, optional
            Whether to create numpy vectors on all processes, or
            just on process 0.

        Returns
        -------
        numpy.ndarray or None
            A numpy array of the vector, or ``None``
            on all processes other than 0 if `to_all == False`.
        '''

        from petsc4py import PETSc

        # scatter seems to be broken for CUDA vectors
        if PETSc.COMM_WORLD.size > 1:
            # collect to process 0
            if to_all:
                sc, v0 = PETSc.Scatter.toAll(vec)
            else:
                sc, v0 = PETSc.Scatter.toZero(vec)

            sc.begin(vec, v0)
            sc.end(vec, v0)

            if not to_all and PETSc.COMM_WORLD.rank != 0:
                return None
        else:
            v0 = vec

        ret = np.ndarray((v0.getSize(),), dtype=np.complex128)
        ret[:] = v0[:]

        return ret

    def to_numpy(self, to_all = False):
        """
        Return a numpy representation of the state.

        Parameters
        ----------
        to_all : bool
            Whether to return the vector on all MPI ranks (True),
            or just rank 0 (False).
        """
        self.assert_initialized()
        return self._to_numpy(self.vec, to_all)

    def _get_nonzero_elements(self):
        '''
        Get a list of the indices and values corresponding to nonzero
        vector elements. If there are more than 10 nonzero elements, only
        the first three and final one are returned, separated by a tuple
        (0, 0) to mark omitted elements. This function is used for the
        pretty printing functions defined below.

        It's complicated because the nonzero elements could be on any MPI rank
        but we want every rank to get the same result.
        '''
        self.assert_initialized()

        # get the functions we need, but without requiring mpi4py
        # if we're running on only 1 rank
        config._initialize()
        from petsc4py import PETSc
        if PETSc.COMM_WORLD.size > 1:
            allgather = PETSc.COMM_WORLD.tompi4py().allgather

            start, end = self.vec.getOwnershipRange()
            local_vec = np.ndarray((end-start,), dtype=np.complex128)
            local_vec[:] = self.vec[start:end]

        else:
            allgather = lambda x: [x]
            start = 0
            local_vec = self.vec

        local_nonzero = np.count_nonzero(local_vec)
        all_nonzero = allgather(local_nonzero)
        if sum(all_nonzero) > 10:
            elements_omitted = True

            # only take the first 3 and final 1
            n_nonzero_preceding = 0
            for rank in range(0, PETSc.COMM_WORLD.rank):
                n_nonzero_preceding += all_nonzero[rank]

            n_nonzero_after = 0
            for rank in range(PETSc.COMM_WORLD.size-1, PETSc.COMM_WORLD.rank, -1):
                n_nonzero_after += all_nonzero[rank]

            local_take_indices = []
            local_nonzero_indices = None

            if n_nonzero_preceding < 3:
                n_local_take = min(3-n_nonzero_preceding, local_nonzero)
                local_nonzero_indices = np.flatnonzero(local_vec)
                local_take_indices += list((start+local_nonzero_indices)[:n_local_take])

            if n_nonzero_after == 0 and local_nonzero:
                # we own the final element
                if local_nonzero_indices is None:
                    local_nonzero_indices = np.flatnonzero(local_vec)

                # we are the last portion with a nonzero element---take it
                local_take_indices.append(start+local_nonzero_indices[-1])

            local_take_values = [self.vec[i] for i in local_take_indices]

            indices = allgather(local_take_indices)
            values = allgather(local_take_values)

        else:
            elements_omitted = False

            # otherwise take everything
            nonzero_local_idxs = np.flatnonzero(local_vec)
            indices = allgather(start+nonzero_local_idxs)
            values = allgather([self.vec[start+i] for i in nonzero_local_idxs])

        # flatten
        indices = [val for ary in indices for val in ary]
        values = [val for ary in values for val in ary]

        if elements_omitted:
            indices.insert(-1, 0)
            values.insert(-1, 0)

        return list(zip(indices, values))

    @classmethod
    def _get_coeff_strs(cls, nonzeros):
        if all(v in (0, 1) for _, v in nonzeros):
            coeffs = ['']*len(nonzeros)
        else:
            if all(v.imag == 0 for _, v in nonzeros):
                format_str = '{v.real:0.3f}'
            else:
                format_str = '({v.real:0.3f}+{v.imag:0.3f}j)'

            coeffs = []
            for _, v in nonzeros:
                if v == 0:
                    coeffs.append('')
                else:
                    coeffs.append(format_str.format(v=v))

        return coeffs

    def _idx_to_str(self, idx):
        int_rep = self.subspace.idx_to_state(idx)[0]

        if self.repr_binary:
            alphabet = '01'
        else:
            alphabet = 'UD'

        rtn = ''
        for i in range(self.L):
            rtn += alphabet[(int_rep >> i) & 1]

        return rtn

    def __str__(self):
        if not self.initialized:
            return '<State with uninitialized contents>'

        nonzeros = self._get_nonzero_elements()

        if not nonzeros:
            return '<zero vector>'

        coeff_strs = self._get_coeff_strs(nonzeros)
        state_strs = []
        for idx, v in nonzeros:
            if v == 0:
                state_strs.append('...')
            else:
                state_strs.append('|'+self._idx_to_str(idx)+'>')

        return ' + '.join(c+s for c, s in zip(coeff_strs, state_strs))

    def _repr_latex_(self):
        if not self.initialized:
            return '<State with uninitialized contents>'

        nonzeros = self._get_nonzero_elements()

        if not nonzeros:
            return '<zero vector>'

        coeff_strs = self._get_coeff_strs(nonzeros)
        state_strs = []
        for idx, v in nonzeros:
            if v == 0:
                state_strs.append(r'\cdots')
            else:
                state_str = self._idx_to_str(idx)
                state_str = state_str.replace('U', r'\uparrow')
                state_str = state_str.replace('D', r'\downarrow')
                state_strs.append(r'\left|'+state_str+r'\right>')

        return '$'+' + '.join(c+s for c, s in zip(coeff_strs, state_strs))+'$'

    def save(self, fname):
        '''
        Save the state to disk. Note that this method saves the state
        as a pair of files, ``<fname>.vec`` and ``<fname>.metadata``.

        Parameters
        ----------

        fname : str
            The path to save the state to
        '''
        self.assert_initialized()

        config._initialize()
        from petsc4py import PETSc

        if PETSc.COMM_WORLD.rank == 0:
            with open(fname+'.metadata', 'wb') as f:
                pickle.dump(self.subspace, f)

        viewer = PETSc.Viewer().createBinary(
            fname+'.vec',
            mode=PETSc.Viewer.Mode.WRITE
        )
        self.vec.view(viewer)

    @classmethod
    def from_file(cls, fname):
        '''
        Load a state from a file. You do not need to create a state to call
        this method---just directly call ``State.from_file(fname)`` and the
        return value will be your new state object.

        .. note::

            This method uses the Python
            ``pickle`` module which is not secure against maliciously constructed
            data; thus this method should not be used on data from untrusted
            sources.

        Parameters
        ----------

        fname : str
            The path from which to load the state

        Returns
        -------

        State
            The state from the file
        '''
        with open(fname+'.metadata', 'rb') as f:
            subspace = pickle.load(f)

        config._initialize()
        from petsc4py import PETSc

        viewer = PETSc.Viewer().createBinary(
            fname+'.vec',
            mode=PETSc.Viewer.Mode.READ
        )
        vec = PETSc.Vec().create()
        vec.load(viewer)

        if subspace.get_dimension() != vec.getSize():
            raise RuntimeError("corrupt data encountered when loading state"
                               "from file")

        rtn = cls(subspace=subspace)
        rtn._vec = vec

        rtn.set_initialized()

        return rtn

    def dot(self, x):
        '''
        Compute the inner product of two states.

        Parameters
        ----------

        x : State
            The state to take the inner product against

        Returns
        -------

        complex
            The value of the inner product
        '''
        self.assert_initialized()
        x.assert_initialized()
        return self.vec.dot(x.vec)

    def norm(self):
        '''
        Compute the Euclidean norm of the state vector.

        Returns
        -------

        float
            The vector norm
        '''
        self.assert_initialized()
        return self.vec.norm()

    def normalize(self):
        '''
        Scale the vector such that the norm is 1.
        '''
        self.assert_initialized()
        return self.vec.normalize()

    def scale(self, c):
        '''
        Scale the vector.

        Parameters
        ----------

        c : float
            The value to scale by
        '''
        self.assert_initialized()
        self.vec.scale(c)

    def __imul__(self, c):
        self.scale(c)
        return self

    def __mul__(self, c):
        rtn = self.copy()
        rtn *= c
        return rtn

    # multiplication is commutative between a
    # vector and a scalar (mat-vec is handled
    # by the Operator class)
    def __rmul__(self, c):
        return self*c

    def __itruediv__(self, c):
        self.scale(1/c)
        return self

    def axpy(self, alpha, x):
        '''
        When ``y.axpy(alpha, x)`` is called, it scales the vector x
        by the scalar alpha, and sums the result into y. In other words,
        it computes ``y = alpha*x + y``.
        '''
        self.scale_and_sum(alpha, 1, x)

    def scale_and_sum(self, alpha, beta, x):
        '''
        Also known as "axpby", when ``y.scale_and_sum(alpha, beta, x)`` is
        called, it computes ``y = alpha*x + beta*y``.
        '''
        self.assert_initialized()
        x.assert_initialized()

        if not self.subspace == x.subspace:
            raise ValueError('subspaces do not match')

        if self.vec is x.vec:
            raise ValueError('x and y cannot be the same State object')

        self.vec.axpby(alpha, beta, x.vec)

    def __iadd__(self, x):
        if isinstance(x, State):
            self.axpy(1.0, x)

        else:  # assume x is a number
            self.assert_initialized()
            self.vec.shift(x)

        return self

    def __add__(self, x):
        rtn = self.copy()
        rtn += x
        return rtn

    def __radd__(self, x):
        return self + x  # all addition to states is commutative

    def __isub__(self, x):
        if isinstance(x, State):
            self.axpy(-1.0, x)

        else:
            self += -x

        return self

    def __sub__(self, x):
        rtn = self.copy()
        rtn -= x
        return rtn

    def __rsub__(self, x):
        rtn = self.copy()
        rtn *= -1
        return rtn + x

    def __len__(self):
        return self.subspace.get_dimension()


class UninitializedError(RuntimeError):
    pass
