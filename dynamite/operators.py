#!/usr/bin/env python3
# ^ that's just for pylinter

"""
This module provides the building blocks for Hamiltonians, and
defines their built-in behavior and operations.
"""

__all__ = ['Sigmax',
           'Sigmay',
           'Sigmaz',
           'Identity',
           'Zero',
           'Sum',
           'Product',
           'IndexSum',
           'IndexProduct']

from . import initialize
initialize()

from itertools import product
import numpy as np

from petsc4py.PETSc import Vec, COMM_WORLD

from .backend.backend import build_mat,destroy_shell_context,MSC_dtype
from .computations import evolve,eigsolve
from ._utils import qtp_identity_product,condense_terms,coeff_to_str,MSC_matrix_product

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

        self._L = L
        self.max_ind = None
        self.needs_parens = False
        self.coeff = 1
        self._mat = None
        self._MSC = None
        self._diag_entries = False
        self._shell = False


    ### computation functions

    def evolve(self,state,t,**kwargs):
        """
        Evolve a state under the Hamiltonian, according to the Schrodinger equation.
        The units are natural; the evolution is:

        .. math::
            \Psi_t = e^{-i H t} \Psi_0

        This class method is just a wrapper on :meth:`dynamite.computations.evolve`.

        .. note:: The spin chain length ``L`` must be set before calling ``evolve``.

        Parameters
        ----------
        state : petsc4py.PETSc.Vec
            A PETSc vector containing the initial state.
            Can be easily created with :meth:`dynamite.tools.build_state`.

        t : float
            The time for which to evolve the state (can be negative to evolve
            backwards in time).

        **kwargs :
            Any further keyword arguments are passed to the underlying call to
            :meth:`dynamite.computations.evolve`. See that documentation for a
            detailed description of possible arguments.

        Returns
        -------
        petsc4py.PETSc.Vec
            The result vector :math:`\Psi_f`.
        """
        return evolve(state,self,t,**kwargs)

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
        numpy.array or tuple(numpy.array, list(petsc4py.PETSc.Vec))
            Either a 1D numpy array of eigenvalues, or a pair containing that array
            and a list of the corresponding eigenvectors.
        """
        return eigsolve(self,**kwargs)

    ### properties

    @property
    def L(self):
        """
        Property defining the length of the spin chain.
        The matrix dimension will then be :math:`d = 2^L`.
        If the length hasn't been set yet, ``L`` is ``None``.
        """
        return self._L

    @L.setter
    def L(self,value):

        if value is not None:
            try:
                valid = int(value) == value
            except ValueError:
                valid = False

            if not (valid and value > 0):
                raise ValueError('L must be a positive integer.')

        if self._mat is not None:
            self.destroy_mat()
            self.release_MSC()
        self._L = value

        # propagate L down the tree of operators
        self._prop_L(value)

    def set_length(self,length):
        """
        Set the length of the spin chain. An alternate interface to :attr:`Operator.L`.

        Equivalent to ``Operator.L = length``.

        Parameters
        ----------
        length : int
            The length of the spin chain.
        """
        self.L = length

    def _prop_L(self,L):
        raise NotImplementedError()

    @property
    def dim(self):
        """
        Read-only attribute returning the dimension :math:`d = 2^L` of the matrix,
        or ``None`` if ``L`` is ``None``.
        """
        if self._L is None:
            return None
        else:
            return 1<<self._L

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
        return self.nnz/self.dim

    @property
    def use_shell(self):
        """
        Switch whether to use shell matrices or not. For a description of shell
        matrices and their benefits, see the documentation.

        .. note::
            Changing this value after the matrix has been built will invoke a call
            to :meth:`Operator.destroy_mat`.
        """
        return self._shell

    @use_shell.setter
    def use_shell(self,value):
        if value != self._shell:
            self.destroy_mat()
        self._shell = value

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
        o = self._copy()
        o.L = self.L
        o.max_ind = self.max_ind
        o.needs_parens = self.needs_parens
        o.coeff = self.coeff
        o._shell = self.use_shell
        return o

    def _copy(self):
        raise NotImplementedError()

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

        if self._shell:
            destroy_shell_context(self._mat)

        self._mat.destroy()
        self._mat = None
        self._shell = None

    def build_mat(self,diag_entries=False):
        """
        Build the PETSc matrix, destroying any matrix that has already been built, and
        store it internally. This function does not return the matrix--see
        :meth:`Operator.get_mat` for that functionality. This function is rarely needed
        by the end user, since it is called automatically whenever the underlying matrix
        needs to be built or rebuilt.
        """
        if self.L is None:
            raise ValueError('Must set number of spins (Operator.L) before building PETSc matrix.')

        self.destroy_mat()

        term_array = self.get_MSC()
        # the thing might be big---should allow it to be freed to give a bit more memory for PETSc
        self.release_MSC()

        if diag_entries and not np.any(term_array['masks'] == 0):
            term_array = np.hstack([np.array([(0,0,0)],dtype=MSC_dtype),term_array])

        if not np.any(term_array['masks'] == 0):
            self._diag_entries = False
        else:
            self._diag_entries = True

        self._mat = build_mat(self.L,
                              np.ascontiguousarray(term_array['masks']),
                              np.ascontiguousarray(term_array['signs']),
                              np.ascontiguousarray(term_array['coeffs']),
                              self.use_shell)

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
        if COMM_WORLD.getRank() != 0:
            return ''

        return '$' + self.build_tex(signs='-') + '$'

    def _get_index_set(self):
        # Get the set of indices in the TeX representation of the object
        raise NotImplementedError()

    def _replace_index(self,ind,rep):
        # Replace an index by some value in the TeX representation
        raise NotImplementedError()

    ### qutip representation of operators

    def build_qutip(self,shift_index=0):
        """
        Build a representation of the operator as a qutip.Qobj. This functionality
        is mostly useful for testing and checking correctness, though also perhaps
        to access some qutip functionality directly from dynamite.

        .. note::
            QuTiP's representation of the matrix will not be split among processes--if
            you call this on a large spin chain, it will build a copy on each process,
            possibly causing your program to run out of memory and crash.

        Parameters
        ----------
        shift_index : int
            Shift the whole operator along the spin chain by ``shift_index`` spins.

        Returns
        -------
        qutip.Qobj
            The operator in qutip representation
        """
        import qutip as qtp

        if self.L is None:
            raise ValueError('Must set L before building qutip representation.')
        return self.coeff * self._build_qutip(shift_index)

    def _build_qutip(self,shift_index):
        raise NotImplementedError()

    ### mask, sign, coefficient representation of operators

    def get_MSC(self,shift_index=None):
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

        Returns
        -------
        numpy.ndarray
            A numpy array containing the representation.
        """
        if self.L is None:
            raise ValueError('Must set L before building term array.')
        if shift_index is None:
            if self._MSC is None:
                self._MSC = self._get_MSC()
            return self._MSC
        else:
            return self._get_MSC(shift_index)

    def _get_MSC(self,shift_index):
        raise NotImplementedError()

    def release_MSC(self):
        """
        Remove the operator's reference to the MSC representation, so that it
        can be garbage collected if there are no other references to it.
        """
        self._MSC = None

    ### unary and binary operations

    def __add__(self,x):
        if isinstance(x,Operator):
            return self._op_add(x)
        else:
            raise TypeError('Addition not supported for types')

    def __sub__(self,x):
        return self + -x

    def __neg__(self):
        return -1*self

    def __mul__(self,x):
        if isinstance(x,Operator):
            return self._op_mul(x)
        elif isinstance(x,Vec):
            return self.get_mat() * x
        else:
            try:
                # check that it is a number, in the
                # most generic way I can think of
                if x == np.array([1]) * x:
                    return self._num_mul(x)
            except (TypeError,ValueError):
                pass
            raise TypeError('Multiplication not supported for types %s and %s' % (str(type(self)),str(type(x))))

    def __rmul__(self,x):
        if isinstance(x,Vec):
            return x*self.get_mat()
        elif any(isinstance(x,t) for t in [float,int,complex]):
            return self.__mul__(x)
        else:
            raise TypeError('Multiplication not supported for types %s and %s' % (str(type(self)),str(type(x))))

    def _op_add(self,o):
        if self.L is not None and o.L is not None and self.L != o.L:
            raise ValueError('Cannot add operators of different sizes (L=%d, L=%d)' % (self.L,o.L))

        if isinstance(o,Sum):
            return Sum(terms = [self] + o.terms)
        elif isinstance(o,Operator):
            return Sum(terms = [self,o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

    def _op_mul(self,o):
        if isinstance(o,Product):
            return Product(terms = [self] + o.terms)
        elif isinstance(o,Operator):
            return Product(terms = [self,o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

    def _num_mul(self,x):
        self.coeff = self.coeff * x
        return self


class _Expression(Operator):
    def __init__(self,terms,L=None):
        """
        Base class for Sum and Product classes.
        """
        Operator.__init__(self,L=L)

        self.terms = [t.copy() for t in terms]

        terms_L = None
        for t in self.terms:
            if t.L is not None:
                if terms_L is not None:
                    if t.L != terms_L:
                        raise ValueError('All terms must have same length L.')
                else:
                    L = t.L

        if len(self.terms) > 1:
            self.max_ind = max(o.max_ind for o in self.terms)
        elif len(self.terms) == 1:
            self.max_ind = self.terms[0].max_ind

        # parent L overrides children -- only set it
        if self.L is None:
            self._L = L
        self.L = self._L # ensure size propagates to all terms

    def _copy(self):
        return type(self)(terms=self.terms)

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

    L : int, optional
        The length of the spin chain
    """

    def __init__(self,terms,L=None):
        _Expression.__init__(self,terms,L=L)
        if len(self.terms) > 1:
            self.needs_parens = True

    def _get_MSC(self,shift_index=0):

        if not self.terms:
            return np.ndarray((0,),dtype=MSC_dtype)

        all_terms = np.hstack([t.get_MSC(shift_index=shift_index) for t in self.terms])
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

    def _build_qutip(self,shift_index):
        ret = Zero(L=self.L).build_qutip(shift_index)

        for term in self.terms:
            ret += term.build_qutip(shift_index)

        return ret

    def _op_add(self,o):
        if isinstance(o,Sum):
            return Sum(terms = self.terms + o.terms)
        elif isinstance(o,Operator):
            return Sum(terms = self.terms + [o])
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

    L : int, optional
        The length of the spin chain
    """

    def __init__(self,terms,L=None):
        _Expression.__init__(self,terms,L=L)
        for term in self.terms:
            self.coeff = self.coeff * term.coeff
            term.coeff = 1

    def _get_MSC(self,shift_index=0):

        if not self.terms:
            return np.ndarray((0,),dtype=MSC_dtype)

        all_terms = MSC_matrix_product(t.get_MSC(shift_index=shift_index) for t in self.terms)
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)

    def _build_tex(self,signs='-',request_parens=False):
        t = coeff_to_str(self.coeff,signs)
        for term in self.terms:
            t += term.build_tex(request_parens=True)
        return t

    def _build_qutip(self,shift_index):
        ret = Identity(L=self.L).build_qutip(shift_index)

        for term in self.terms:
            ret *= term.build_qutip(shift_index)

        return ret

    def _op_mul(self,o):
        if isinstance(o,Product):
            return Product(terms = self.terms + o.terms)
        elif isinstance(o,Operator):
            return Product(terms = self.terms + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class _IndexType(Operator):

    def __init__(self,op,min_i=0,max_i=None,index_label='i',wrap=False,L=None):

        self.min_i = min_i
        self.max_i = max_i
        self.wrap = wrap

        Operator.__init__(self,L=L)

        if self.max_i is not None and self.max_i < self.min_i:
            raise IndexError('max_i must be >= min_i.')

        if self.min_i < 0:
            raise IndexError('min_i must be >= 0.')

        self.max_ind = op.max_ind
        if not wrap and (self.max_i is not None and self.L is not None and
                         self.max_i >= (self.L - self.max_ind)):
            raise IndexError('max_i would cause operators to extend past '
                             'end of spin chain and wrap == False.')

        self.op = op.copy()

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

        if self.op.L is not None and self.L is None:
            self._L = self.op.L
        self.L = self._L # propagate L

    def _copy(self):
        o = type(self)(op=self.op,
                       min_i=self.min_i,
                       max_i=self.max_i,
                       index_label=self.index_label,
                       wrap=self.wrap)
        return o

    def _get_sigma_tex(self):
        raise NotImplementedError()

    def _build_tex(self,signs='-',request_parens=False):
        t = ''
        if request_parens:
            t += r'\left['
        t += coeff_to_str(self.coeff,signs)
        t += self._get_sigma_tex()+r'_{'+self.index_label+'='+str(self.min_i)+'}'
        if self.max_i is not None:
            t += '^{'+str(self.max_i)+'}'
        else:
            if self.wrap:
                t += '^{L-1}'
            else:
                t += '^{L'+('-'+str(self.max_ind+1))+'}'
        t += self.op.build_tex(request_parens=True)
        if request_parens:
            t += r'\right]'
        return t

    def _get_index_set(self):
        return self.op._get_index_set()

    def _replace_index(self,ind,rep):
        return self.op._replace_index(ind,rep)

    def _prop_L(self,L):
        if self.max_i is not None and L is not None:
            if self.max_i > L-1:
                raise IndexError('Cannot set L smaller than '
                                 'max_i+1 of index sum or product.')
            elif not self.wrap and self.max_i + self.max_ind > L-1:
                raise IndexError('Index sum or product operator would extend '
                                 'past end of spin chain (L too small).')

        self.op.L = L
        if self.max_i is None and L is not None:
            if self.wrap:
                self.max_i = L-1
            else:
                self.max_i = L - self.max_ind - 1

    def _get_MSC(self,shift_index=0):
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

    def _get_MSC(self,shift_index=0):
        all_terms = np.hstack([self.op.get_MSC(shift_index=shift_index+i) for i in range(self.min_i,self.max_i+1)])
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)

    def _build_qutip(self,shift_index):
        ret = Zero(L=self.L).build_qutip()

        for i in range(self.min_i,self.max_i+1):
            ret += self.op.build_qutip(shift_index=i)

        return ret

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

    def _get_MSC(self,shift_index=0):

        terms = (self.op.get_MSC(shift_index=shift_index+i) for i in range(self.min_i,self.max_i+1))
        all_terms = MSC_matrix_product(terms)
        all_terms['coeffs'] *= self.coeff
        return condense_terms(all_terms)

    def _build_qutip(self,shift_index):
        ret = Identity(L=self.L).build_qutip()

        for i in range(self.min_i,self.max_i+1):
            ret *= self.op.build_qutip(shift_index=i)

        return ret

# the bottom level. a single operator (e.g. sigmax)
class _Fundamental(Operator):

    def __init__(self,index=0,L=None):

        Operator.__init__(self,L=L)
        self.index = index
        self.max_ind = index
        self.tex = []
        self.tex_end = ''

    def _copy(self):
        return type(self)(index=self.index)

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

    def _get_MSC(self,shift_index=0):
        raise NotImplementedError()

class Sigmax(_Fundamental):
    """
    The Pauli :math:`\sigma_x` operator on site :math:`i`.
    """

    def __init__(self,index=0,L=None):
        _Fundamental.__init__(self,index,L=L)
        self.tex = [[r'\sigma^x_{',self.index]]
        self.tex_end = r'}'

    def _get_MSC(self,shift_index=0):
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')
        return np.array([(1<<ind,0,self.coeff)],dtype=MSC_dtype)

    def _build_qutip(self,shift_index):
        import qutip as qtp
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.sigmax(),ind,self.L)

class Sigmaz(_Fundamental):
    """
    The Pauli :math:`\sigma_z` operator on site :math:`i`.
    """

    def __init__(self,index=0,L=None):
        _Fundamental.__init__(self,index,L=L)
        self.tex = [[r'\sigma^z_{',self.index]]
        self.tex_end = r'}'

    def _get_MSC(self,shift_index=0):
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')
        return np.array([(0,1<<ind,self.coeff)],dtype=MSC_dtype)

    def _build_qutip(self,shift_index):
        import qutip as qtp
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.sigmaz(),ind,self.L)

class Sigmay(_Fundamental):
    """
    The Pauli :math:`\sigma_y` operator on site :math:`i`.
    """

    def __init__(self,index=0,L=None):
        _Fundamental.__init__(self,index,L=L)
        self.tex = [[r'\sigma^y_{',self.index]]
        self.tex_end = r'}'

    def _get_MSC(self,shift_index=0):
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')
        return np.array([(1<<ind,1<<ind,-1j*self.coeff)],dtype=MSC_dtype)

    def _build_qutip(self,shift_index):
        import qutip as qtp
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.sigmay(),ind,self.L)

class Identity(_Fundamental):
    """
    The identity operator. Since it is tensored with identities on all
    the rest of the sites, the ``index`` argument has no effect.
    """

    def __init__(self,index=0,L=None):
        _Fundamental.__init__(self,index,L=L)
        self.tex = []
        self.tex_end = r'I'
        self.max_ind = 0

    def _get_MSC(self,shift_index=0):
        return np.array([(0,0,self.coeff)],dtype=MSC_dtype)

    def _build_qutip(self,shift_index):
        import qutip as qtp
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.identity(2),ind,self.L)

# also should hide this tex when appropriate
class Zero(_Fundamental):
    """
    The zero operator---equivalent to a matrix of all zeros of dimension :math:`2^L`.
    Like for the identity, the ``index`` argument has no effect.
    """

    def __init__(self,index=0,L=None):
        _Fundamental.__init__(self,index,L=L)
        self.tex = []
        self.tex_end = r'0'
        self.max_ind = 0

    def _get_MSC(self,shift_index=0):
        return np.array([(0,0,0)],dtype=MSC_dtype)

    def _build_qutip(self,shift_index):
        import qutip as qtp
        ind = self.index+shift_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.Qobj([[0,0],[0,0]]),ind,self.L)
