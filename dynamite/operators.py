#!/usr/bin/python3

from . import initialize
initialize()

from itertools import product
from copy import deepcopy
import numpy as np

try:
    import qutip as qtp
except ImportError:
    qtp = None

from petsc4py.PETSc import Vec, COMM_WORLD

from .backend.backend import build_mat,destroy_shell_context
from .computations import evolve,eigsolve
from .utils import product_of_terms,term_dtype,qtp_identity_product

class Operator:

    def __init__(self,L=None):

        self.L = L
        self.max_ind = None
        self.needs_parens = False
        self.coeff = 1
        self._mat = None
        self._diag_entries = False
        self.is_shell = None

    def set_size(self,L):
        self.L = L
        if self._mat is not None:
            self.destroy_mat()
        self._set_size(L)

    def _set_size(self,L):
        raise NotImplementedError()

    def evolve(self,x,*args,**kwargs):
        return evolve(x,self,*args,**kwargs)

    def eigsolve(self,*args,**kwargs):
        return eigsolve(self,*args,**kwargs)

    @classmethod
    def condense_terms(cls,all_terms):

        all_terms.sort(order=['masks','signs'])

        combined = np.ndarray((len(all_terms),),dtype=term_dtype())

        i = 0
        n = 0
        maxn = len(all_terms)
        while n < maxn:
            t = all_terms[n]
            combined[i] = t
            n += 1
            while n < maxn and all_terms[n]['masks'] == t['masks'] and all_terms[n]['signs'] == t['signs']:
                combined[i]['coeffs'] += all_terms[n]['coeffs']
                n += 1
            i += 1

        combined.resize((i,)) # get rid of the memory we don't need

        return combined

    def build_mat(self,shell=False,diag_entries=False):
        if self.L is None:
            raise ValueError('Must set number of spins (Operator.set_size(L)) before building PETSc matrix.')

        if self._mat is not None:
            # destroy the old one
            self._mat.destroy()

        term_array = self.build_term_array()

        if diag_entries and not np.any(term_array['masks'] == 0):
            term_array = np.hstack([np.array([(0,0,0)],dtype=term_dtype()),term_array])
            print(term_array)

        self._mat = build_mat(self.L,
                              np.ascontiguousarray(term_array['masks']),
                              np.ascontiguousarray(term_array['signs']),
                              np.ascontiguousarray(term_array['coeffs']),
                              shell)

        self.is_shell = shell

    def destroy_mat(self):

        if self._mat is None:
            return

        if self.is_shell:
            destroy_shell_context(self._mat)

        self._mat.destroy()
        self._mat = None
        self.is_shell = None

    def get_mat(self,diag_entries=False):
        if self._mat is None or self._diag_entries != diag_entries:
            self.build_mat(diag_entries=diag_entries)
            self._diag_entries = diag_entries
        return self._mat

    @classmethod
    def coeff_to_str(cls,x,signs='+-'):
        if x == 1:
            return '+' if '+' in signs else ''
        elif x == -1:
            return '-' if '-' in signs else ''
        else:
            return ('+' if '+' in signs else '')+str(x)

    def _repr_latex_(self):

        # don't print if we aren't process 0
        # this causes there to still be some awkward white space,
        # but at least it doesn't print the tex a zillion times
        if COMM_WORLD.getRank() != 0:
            return ''

        return '$' + self.build_tex(signs='-') + '$'

    def build_tex(self,signs='-',request_parens=False):
        return self._build_tex(signs,request_parens)

    def _build_tex(self,signs='-',request_parens=False):
        raise NotImplementedError()

    def get_index_set(self):
        raise NotImplementedError()

    def replace_index(self,ind,rep):
        raise NotImplementedError()

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
        elif any(isinstance(x,t) for t in [float,int,complex]): # TODO: this is bad. find a better way to check if it's a number
            return self._num_mul(x)
        elif isinstance(x,Vec):
            return self.get_mat() * x
        else:
            raise TypeError('Multiplication not supported for types %s and %s' % (str(type(self)),str(type(x))))

    def __rmul__(self,x):
        if isinstance(x,Vec):
            return x*self.get_mat()
        elif any(isinstance(x,t) for t in [float,int,complex]):
            return self.__mul__(x)
        else:
            raise TypeError('Multiplication not supported for types %s and %s' % (str(type(self)),str(type(x))))

    def build_term_array(self,add_index=0):
        if self.L is None:
            raise ValueError('Must set L before building term array.')
        return self._build_term_array(add_index)

    def _build_term_array(self,add_index):
        raise NotImplementedError()

    def build_qutip(self,add_index=0):
        if qtp is None:
            raise ImportError('Could not import qutip.')
        if self.L is None:
            raise ValueError('Must set L before building qutip representation.')
        return self.coeff * self._build_qutip(add_index)

    def _build_qutip(self,add_index):
        raise NotImplementedError()

    def _op_add(self,o):
        if self.L is not None and o.L is not None and self.L != o.L:
            raise ValueError('Cannot add operators of different sizes (L=%d, L=%d)' % (self.L,o.L))

        if isinstance(o,SumTerms):
            return SumTerms(terms = [self] + o.terms)
        elif isinstance(o,Operator):
            return SumTerms(terms = [self,o])
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


class Expression(Operator):
    def __init__(self,terms,**kwargs):
        Operator.__init__(self,**kwargs)

        terms = list(terms)

        for t in terms:
            # make sure we don't copy a huge matrix...
            # TODO: this could confuse people re: shell matrices (if they call build_mat themselves)
            # should define a copy function
            t.destroy_mat()
            L = None
            if t.L is not None:
                if L is not None:
                    if t.L != L:
                        raise ValueError('All terms must have same length L.')
                else:
                    L = t.L

        # parent L overrides children
        if self.L is None:
            self.L = L

        terms = deepcopy(list(terms))

        self.terms = terms
        if len(self.terms) > 1:
            self.max_ind = max(o.max_ind for o in self.terms)
        elif len(self.terms) == 1:
            self.max_ind = self.terms[0].max_ind

        self.set_size(self.L) # ensure size propagates to all terms

    def get_index_set(self):
        indices = set()
        for term in self.terms:
            indices = indices | term.get_index_set()
        return indices

    def replace_index(self,ind,rep):
        for term in self.terms:
            term.replace_index(ind,rep)

    def _build_tex(self,signs='-',request_parens=False):
        raise NotImplementedError()

    def _set_size(self,L):
        for term in self.terms:
            term.set_size(L)

class SumTerms(Expression):

    def __init__(self,terms,**kwargs):
        Expression.__init__(self,terms,**kwargs)
        if len(self.terms) > 1:
            self.needs_parens = True

    def _build_term_array(self,add_index=0):

        if not self.terms:
            return np.ndarray((0,),dtype=term_dtype())

        all_terms = np.hstack([t.build_term_array(add_index=add_index) for t in self.terms])
        all_terms['coeffs'] *= self.coeff
        return self.condense_terms(all_terms)

    def _build_tex(self,signs='-',request_parens=False):
        t = self.coeff_to_str(self.coeff,signs)
        add_parens = False
        if (t and self.needs_parens) or (request_parens and self.needs_parens):
            add_parens = True
            t += r'\left('
        for n,term in enumerate(self.terms):
            t += term.build_tex(signs=('+-' if n else signs))
        if add_parens:
            t += r'\right)'
        return t

    def _build_qutip(self,add_index):
        ret = Zero(L=self.L).build_qutip(add_index)

        for term in self.terms:
            ret += term.build_qutip(add_index)

        return ret

    def _op_add(self,o):
        if isinstance(o,SumTerms):
            return SumTerms(terms = self.terms + o.terms)
        elif isinstance(o,Operator):
            return SumTerms(terms = self.terms + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class Product(Expression):

    def __init__(self,terms,**kwargs):
        Expression.__init__(self,terms,**kwargs)
        for term in self.terms:
            self.coeff = self.coeff * term.coeff
            term.coeff = 1

    def _build_term_array(self,add_index=0):

        if not self.terms:
            return np.ndarray((0,),dtype=term_dtype())

        arrays = [t.build_term_array(add_index=add_index) for t in self.terms]

        sizes = np.array([a.shape[0] for a in arrays])
        all_terms = np.ndarray((np.prod(sizes),),dtype=term_dtype())

        prod_terms = product(*arrays)

        for n,t in enumerate(prod_terms):
            all_terms[n] = product_of_terms(t)
            all_terms[n]['coeffs'] *= self.coeff

        return self.condense_terms(all_terms)

    def _build_tex(self,signs='-',request_parens=False):
        t = self.coeff_to_str(self.coeff,signs)
        for term in self.terms:
            t += term.build_tex(request_parens=True)
        return t

    def _build_qutip(self,add_index):
        ret = Identity(L=self.L).build_qutip(add_index)

        for term in self.terms:
            ret *= term.build_qutip(add_index)

        return ret

    def _op_mul(self,o):
        if isinstance(o,Product):
            return Product(terms = self.terms + o.terms)
        elif isinstance(o,Operator):
            return Product(terms = self.terms + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class SigmaType(Operator):

    def __init__(self,op,**kwargs):

        self.min_i = kwargs.pop('min_i',0)
        self.max_i = kwargs.pop('max_i',None)
        index_label = kwargs.pop('index_label','i')

        Operator.__init__(self,**kwargs)

        if self.max_i is not None and self.max_i < self.min_i:
            raise IndexError('max_i must be >= min_i.')

        if self.min_i < 0:
            raise IndexError('min_i must be >= 0.')

        if self.max_i is not None and self.L is not None and self.max_i > self.L:
            raise IndexError('max_i must be <= the size L.')

        # TODO: see above about this. find a better way to not copy a big matrix
        op.destroy_mat()
        self.op = deepcopy(op)

        self.max_ind = self.op.max_ind
        if not isinstance(index_label,str):
            raise TypeError('Index label should be a string.')
        self.index_label = index_label

        indices = self.op.get_index_set()
        if any(not isinstance(x,int) for x in indices):
            raise TypeError('Can only sum/product over objects with integer indices')
        if min(indices) != 0:
            raise IndexError('Minimum index of summand must be 0.')

        for ind in indices:
            if isinstance(ind,int):
                self.op.replace_index(ind,index_label+'+'+str(ind) if ind else index_label)

        if self.op.L is not None and self.L is None:
            self.L = self.op.L
        self.set_size(self.L) # propagate L

    def get_sigma_tex(self):
        raise NotImplementedError()

    def _build_tex(self,signs='-',request_parens=False):
        t = ''
        if request_parens:
            t += r'\left['
        t += self.coeff_to_str(self.coeff,signs)
        t += self.get_sigma_tex()+r'_{'+self.index_label+'='+str(self.min_i)+'}'
        if self.max_i is not None:
            t += '^{'+str(self.max_i)+'}'
        else:
            t += '^{L'+('-'+str(self.max_ind+1))+'}'
        t += self.op.build_tex(request_parens=True)
        if request_parens:
            t += r'\right]'
        return t

    def get_index_set(self):
        return self.op.get_index_set()

    def replace_index(self,ind,rep):
        return self.op.replace_index(ind,rep)

    def _set_size(self,L):
        if self.max_i is not None and L is not None and self.max_i > L:
            raise IndexError('Cannot set L smaller than max_i of PiProd.')
        self.op.set_size(L)
        if self.max_i is None and L is not None:
            self.max_i = L - self.max_ind - 1

    def _build_term_array(self,add_index=0):
        raise NotImplementedError()

class SigmaSum(SigmaType):
    def __init__(self,op,**kwargs):
        SigmaType.__init__(self,op,**kwargs)

    def get_sigma_tex(self):
        return r'\sum'

    def _build_term_array(self,add_index=0):
        all_terms = np.hstack([self.op.build_term_array(add_index=add_index+i) for i in range(self.min_i,self.max_i+1)])
        all_terms['coeffs'] *= self.coeff
        return self.condense_terms(all_terms)

    def _build_qutip(self,add_index):
        ret = Zero(L=self.L).build_qutip()

        for i in range(self.min_i,self.max_i+1):
            ret += self.op.build_qutip(add_index=i)

        return ret

class PiProd(SigmaType):
    def __init__(self,op,**kwargs):
        SigmaType.__init__(self,op,**kwargs)

    def get_sigma_tex(self):
        return r'\prod'

    def _build_term_array(self,add_index=0):

        arrays = [self.op.build_term_array(add_index=add_index+i) for i in range(self.min_i,self.max_i+1)]
        nfacs = self.max_i - self.min_i + 1
        all_terms = np.ndarray((arrays[0].shape[0]**nfacs,),dtype=term_dtype())

        prod_terms = product(*arrays)
        for n,t in enumerate(prod_terms):
            all_terms[n] = product_of_terms(t)
            all_terms[n]['coeffs'] *= self.coeff

        return self.condense_terms(all_terms)

    def _build_qutip(self,add_index):
        ret = Identity(L=self.L).build_qutip()

        for i in range(self.min_i,self.max_i+1):
            ret *= self.op.build_qutip(add_index=i)

        return ret

# the bottom level. a single operator (e.g. sigmax)
class Fundamental(Operator):

    def __init__(self,index=0,**kwargs):

        Operator.__init__(self,**kwargs)
        self.index = index
        self.max_ind = index
        self.tex = []
        self.tex_end = ''

    def get_index_set(self):
        indices = set()
        for t in self.tex:
            indices = indices | {t[1]}
        return indices

    def replace_index(self,ind,rep):
        for t in self.tex:
            if t[1] == ind:
                t[1] = rep

    def _build_tex(self,signs='-',request_parens=False):
        t = self.coeff_to_str(self.coeff,signs)
        for substring,index in self.tex:
            t += substring + str(index)
        t += self.tex_end
        return t

    def _set_size(self,L):
        pass

    def _build_term_array(self,add_index=0):
        raise NotImplementedError()

class Sigmax(Fundamental):
    def __init__(self,index=0,**kwargs):
        Fundamental.__init__(self,index,**kwargs)
        self.tex = [[r'\sigma_x^{',self.index]]
        self.tex_end = r'}'

    def _build_term_array(self,add_index=0):
        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')
        return np.array([(1<<ind,0,self.coeff)],dtype=term_dtype())

    def _build_qutip(self,add_index):

        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.sigmax(),ind,self.L)

class Sigmaz(Fundamental):
    def __init__(self,index=0,**kwargs):
        Fundamental.__init__(self,index,**kwargs)
        self.tex = [[r'\sigma_z^{',self.index]]
        self.tex_end = r'}'

    def _build_term_array(self,add_index=0):
        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')
        return np.array([(0,1<<ind,self.coeff)],dtype=term_dtype())

    def _build_qutip(self,add_index):

        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.sigmaz(),ind,self.L)

class Sigmay(Fundamental):
    def __init__(self,index=0,**kwargs):
        Fundamental.__init__(self,index,**kwargs)
        self.tex = [[r'\sigma_y^{',self.index]]
        self.tex_end = r'}'

    def _build_term_array(self,add_index=0):
        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')
        return np.array([(1<<ind,1<<ind,-1j*self.coeff)],dtype=term_dtype())

    def _build_qutip(self,add_index):

        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.sigmay(),ind,self.L)

# TODO: should hide the tex if we multiply by something else...
class Identity(Fundamental):
    def __init__(self,index=0,**kwargs):
        Fundamental.__init__(self,index,**kwargs)
        self.tex = []
        self.tex_end = r'I'
        self.max_ind = 0

    def _build_term_array(self,add_index=0):
        return np.array([(0,0,self.coeff)],dtype=term_dtype())

    def _build_qutip(self,add_index):

        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.identity(2),ind,self.L)

# also should hide this tex when appropriate
class Zero(Fundamental):
    def __init__(self,index=0,**kwargs):
        Fundamental.__init__(self,index,**kwargs)
        self.tex = []
        self.tex_end = r'0'
        self.max_ind = 0

    def _build_term_array(self,add_index=0):
        return np.array([(0,0,0)],dtype=term_dtype())

    def _build_qutip(self,add_index):

        ind = self.index+add_index
        if ind >= self.L:
            raise IndexError('requested too large an index')

        return qtp_identity_product(qtp.Qobj([[0,0],[0,0]]),ind,self.L)
