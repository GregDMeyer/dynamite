#!/usr/bin/python3

from itertools import product
from copy import deepcopy
import numpy as np
import atexit
from .backend.backend import build_mat,destroy_shell_context
from .computations import mgr,evolve

from petsc4py.PETSc import Vec

class Operator:

    def __init__(self):

        self.L = None
        self.max_ind = None
        self.needs_parens = False
        self.coeff = 1
        self._mat = None
        self.is_shell = None
        self._dag = False # whether to take the complex conjugate

    def _set_size(self,L):
        self.L = L
        if self._mat is not None:
            self.destroy_mat()

    def set_size(self,L):
        raise NotImplementedError()

    def evolve(self,x,t=None,result=None,tol=None,mfn=None):
        return evolve(x,self,t,result,tol,mfn)

    def dag(self):
        self.destroy_mat()
        o = deepcopy(self)
        o._dag = True
        return o

    @classmethod
    def term_dtype(cls):
        return np.dtype([('masks',np.int32),('signs',np.int32),('coeffs',np.complex128)])

    @classmethod
    def condense_terms(cls,all_terms):

        all_terms.sort(order=['masks','signs'])

        combined = np.ndarray((len(all_terms),),dtype=cls.term_dtype())

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

    def build_mat(self,shell=False):
        if self.L is None:
            raise Exception('Must set number of spins (Operator.set_size(L)) before building PETSc matrix.')

        mgr.initialize_slepc()

        if self._mat is not None:
            # destroy the old one
            self._mat.destroy()

        # TODO: don't forget--need to initialize slepc!
        term_array = self.build_term_array()

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

    def get_mat(self):
        if self._mat is None:
            self.build_mat()
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
        return '$' + self.build_tex('-') + '$'

    def build_tex(self,signs='-',request_parens=False):
        t = self._build_tex(signs,request_parens)

        # # crap. this is a problem if there is already an index...
        # if self._dag:
        #     t += '^{\dagger}'

        return t

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
        a = self._build_term_array(add_index)
        if self._dag:
            a['coeffs'] = np.conj(a['coeffs'])
        return a

    def _build_term_array(self,add_index):
        raise NotImplementedError()

    def _op_add(self,o):
        if self.L is not None and o.L is not None and self.L != o.L:
            raise Exception('Cannot add operators of different sizes (L=%d, L=%d)' % (self.L,o.L))

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
    def __init__(self,terms = None):
        Operator.__init__(self)
        if terms is None:
            terms = []
        else:
            for t in terms:
                # make sure we don't copy a huge matrix...
                # TODO: this could confuse people re: shell matrices (if they call build_mat themselves)
                # should define a copy function
                t.destroy_mat()
            terms = deepcopy(list(terms))
        self.terms = terms
        if len(self.terms) > 1:
            self.max_ind = max(o.max_ind for o in self.terms)
        elif len(self.terms) == 1:
            self.max_ind = self.terms[0].max_ind

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

    def set_size(self,L):
        self._set_size(L)
        for term in self.terms:
            term.set_size(L)

class SumTerms(Expression):

    def __init__(self,terms = None):
        Expression.__init__(self,terms)
        if len(terms) > 1:
            self.needs_parens = True

    def _build_term_array(self,add_index=0):

        if not self.terms:
            return np.ndarray((0,),dtype=Operator.term_dtype())

        all_terms = np.hstack([t.build_term_array(add_index=add_index) for t in self.terms])
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

    def _op_add(self,o):
        if isinstance(o,SumTerms):
            return SumTerms(terms = self.terms + o.terms)
        elif isinstance(o,Operator):
            return SumTerms(terms = self.terms + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class Product(Expression):

    def __init__(self,terms = None):
        Expression.__init__(self,terms)
        for term in self.terms:
            self.coeff = self.coeff * term.coeff
            term.coeff = 1

    def _build_term_array(self,add_index=0):

        if not self.terms:
            return np.ndarray((0,),dtype=Operator.term_dtype())

        arrays = [t.build_term_array(add_index=add_index) for t in self.terms]

        sizes = np.array([a.shape[0] for a in arrays])
        all_terms = np.ndarray((np.prod(sizes),),dtype=Operator.term_dtype())

        prod_terms = product(*arrays)

        for n,t in enumerate(prod_terms):
            prod = np.array([(0,0,1)],dtype=Operator.term_dtype())
            for factor in t:
                prod['masks'] = prod['masks'] ^ factor['masks']
                prod['signs'] = prod['signs'] ^ factor['signs']
                prod['coeffs'] *= factor['coeffs']
            prod['coeffs'] *= self.coeff
            all_terms[n] = prod

        return self.condense_terms(all_terms)

    def _build_tex(self,signs='-',request_parens=False):
        t = self.coeff_to_str(self.coeff,signs)
        for term in self.terms:
            t += term.build_tex(request_parens=True)
        return t

    def _op_mul(self,o):
        if isinstance(o,Product):
            return Product(terms = self.terms + o.terms)
        elif isinstance(o,Operator):
            return Product(terms = self.terms + [o])
        else:
            raise TypeError('Cannot sum expression with type '+type(o))

class SigmaType(Operator):

    def __init__(self,op,min_i=0,max_i=None,index_label='i'):
        Operator.__init__(self)
        self.min_i = min_i
        self.max_i = max_i

        # TODO: see above about this. find a better way to not copy a big matrix
        op.destroy_mat()
        self.op = deepcopy(op)

        self.max_ind = self.op.max_ind
        if not isinstance(index_label,str):
            raise Exception('Index label should be a string.')
        self.index_label = index_label

        indices = self.op.get_index_set()
        if any(not isinstance(x,int) for x in indices):
            raise TypeError('Can only sum/product over objects with integer indices')
        if min(indices) != 0:
            raise TypeError('Minimum index of summand must be 0.')

        for ind in indices:
            if isinstance(ind,int):
                self.op.replace_index(ind,index_label+'+'+str(ind) if ind else index_label)

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

    def set_size(self,L):
        self._set_size(L)
        self.op.set_size(L)
        if self.max_i is None:
            self.max_i = L - self.max_ind - 1

    def _build_term_array(self,add_index=0):
        raise NotImplementedError()

class SigmaSum(SigmaType):
    def __init__(self,*args,**kwargs):
        SigmaType.__init__(self,*args,**kwargs)

    def get_sigma_tex(self):
        return r'\sum'

    def _build_term_array(self,add_index=0):
        all_terms = np.hstack([self.op.build_term_array(add_index=add_index+i) for i in range(self.min_i,self.max_i+1)])
        return self.condense_terms(all_terms)

class PiProd(SigmaType):
    def __init__(self,*args,**kwargs):
        SigmaType.__init__(self,*args,**kwargs)

    def get_sigma_tex(self):
        return r'\prod'

    def _build_term_array(self,add_index=0):

        arrays = [self.op.build_term_array(add_index=add_index+i) for i in range(self.min_i,self.max_i+1)]
        nfacs = self.max_i - self.min_i + 1
        all_terms = np.ndarray((arrays[0].shape[0]**nfacs,),dtype=Operator.term_dtype())

        prod_terms = product(*arrays)
        for n,t in enumerate(prod_terms):
            prod = np.array([(0,0,1)],dtype=Operator.term_dtype())
            for factor in t:
                prod['masks'] = prod['masks'] ^ factor['masks']
                prod['signs'] = prod['signs'] ^ factor['signs']
                prod['coeffs'] *= factor['coeffs']
            prod['coeffs'] *= self.coeff
            all_terms[n] = prod

        return self.condense_terms(all_terms)

# the bottom level. a single operator (e.g. sigmax)
class Fundamental(Operator):

    def __init__(self):
        Operator.__init__(self)
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

    def set_size(self,L):
        self._set_size(L)

    def _build_term_array(self):
        raise NotImplementedError()

class Sigmax(Fundamental):
    def __init__(self,index=0):
        Fundamental.__init__(self)
        self.index = index
        self.tex = [[r'\sigma_x^{',index]]
        self.tex_end = r'}'
        self.max_ind = index

    def _build_term_array(self,add_index=0):
        ind = self.index+add_index
        if ind >= self.L:
            raise Exception('requested too large an index')
        return np.array([(1<<ind,0,self.coeff)],dtype=self.term_dtype())

class Sigmaz(Fundamental):
    def __init__(self,index=0):
        Fundamental.__init__(self)
        self.index = index
        self.tex = [[r'\sigma_z^{',index]]
        self.tex_end = r'}'
        self.max_ind = index

    def _build_term_array(self,add_index=0):
        ind = self.index+add_index
        if ind >= self.L:
            raise Exception('requested too large an index')
        return np.array([(0,1<<ind,self.coeff)],dtype=self.term_dtype())

class Sigmay(Fundamental):
    def __init__(self,index=0):
        Fundamental.__init__(self)
        self.index = index
        self.tex = [[r'\sigma_y^{',index]]
        self.tex_end = r'}'
        self.max_ind = index

    def _build_term_array(self,add_index=0):
        ind = self.index+add_index
        if ind >= self.L:
            raise Exception('requested too large an index')
        return np.array([(1<<ind,1<<ind,1j*self.coeff)],dtype=self.term_dtype())

# TODO: should hide the tex if we multiply by something else...
class Identity(Fundamental):
    def __init__(self):
        Fundamental.__init__(self)
        self.tex = []
        self.tex_end = r'I'
        self.max_ind = 0

    def _build_term_array(self,add_index=0):
        return np.array([(0,0,self.coeff)],dtype=self.term_dtype())

# also should hide this tex when appropriate
class Zero(Fundamental):
    def __init__(self):
        Fundamental.__init__(self)
        self.tex = []
        self.tex_end = r'0'
        self.max_ind = 0

    def _build_term_array(self,add_index=0):
        return np.array([(0,0,0)],dtype=self.term_dtype())