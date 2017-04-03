#!/usr/bin/python3

from itertools import product
from copy import deepcopy
from .backend import build_mat,destroy_shell_context
import numpy as np

class Operator:

    def __init__(self):

        self.L = None
        self.max_ind = None
        self.needs_parens = False
        self.coeff = 1
        self.mat = None
        self.is_shell = None

    def set_size(self,L):
        raise NotImplementedError()

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
        if self.mat is not None:
            # destroy the old one
            self.mat.destroy()

        # TODO: don't forget--need to initialize slepc!
        term_array = self.build_term_array()

        self.mat = build_mat(self.L,
                             np.ascontiguousarray(term_array['masks']),
                             np.ascontiguousarray(term_array['signs']),
                             np.ascontiguousarray(term_array['coeffs']),
                             shell)

        self.is_shell = shell

    def destroy_mat(self):
        if self.is_shell:
            destroy_shell_context(self.mat)

        self.mat.destroy()
        self.mat = None
        self.is_shell = None

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

    def build_tex(self,signs='+-',request_parens=False):
        # this is done by derived classes
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
        else:
            raise TypeError('Multiplication not supported for types')

    def __rmul__(self,x):
        ''' This should only be called for number multiplication '''
        return self.__mul__(x)

    def build_term_array(self):
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
            terms = deepcopy(terms)
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

    def build_tex(self,signs='-',request_parens=False):
        raise NotImplementedError()

    def set_size(self,L):
        self.L = L
        for term in self.terms:
            term.set_size(L)

class SumTerms(Expression):

    def __init__(self,terms = None):
        Expression.__init__(self,terms)
        if len(terms) > 1:
            self.needs_parens = True

    def build_term_array(self):

        if not self.terms:
            return np.ndarray((0,),dtype=Operator.term_dtype())

        all_terms = np.hstack([t.build_term_array() for t in self.terms])
        return self.condense_terms(all_terms)

    def build_tex(self,signs='-',request_parens=False):
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

    def build_term_array(self):

        if not self.terms:
            return np.ndarray((0,),dtype=Operator.term_dtype())

        arrays = [t.build_term_array() for t in self.terms]

        sizes = np.array([a.shape[0] for a in arrays])
        all_terms = np.ndarray((np.prod(sizes),),dtype=Operator.term_dtype())

        prod_terms = product(arrays)
        for n,t in enumerate(prod_terms):
            prod = np.array([(0,0,1)],dtype=Operator.term_dtype())
            for factor in t:
                prod['mask'] = prod['mask'] ^ factor['mask']
                prod['sign'] = prod['sign'] ^ factor['sign']
                prod['coeff'] *= factor['coeff']
            prod['coeff'] *= self.coeff
            all_terms[n] = prod

        return self.condense_terms(all_terms)

    def build_tex(self,signs='-',request_parens=False):
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

# TODO: implement build_term_array for these!
class SigmaType(Operator):

    def __init__(self,op,index_label='i'):
        Operator.__init__(self)
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

    def build_tex(self,signs='-',request_parens=False):
        t = ''
        if request_parens:
            t += r'\left['
        t += self.coeff_to_str(self.coeff,signs)
        t += self.get_sigma_tex()+r'_{'+self.index_label+'=0}'
        if self.L is not None:
            t += '^{'+str(self.L-self.max_ind-1)+'}'
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
        self.L = L
        self.op.set_size(L)

class SigmaSum(SigmaType):
    def __init__(self,*args,**kwargs):
        SigmaType.__init__(self,*args,**kwargs)

    def get_sigma_tex(self):
        return r'\sum'

class PiProd(SigmaType):
    def __init__(self,*args,**kwargs):
        SigmaType.__init__(self,*args,**kwargs)

    def get_sigma_tex(self):
        return r'\prod'

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

    def build_tex(self,signs='-',request_parens=False):
        t = self.coeff_to_str(self.coeff,signs)
        for substring,index in self.tex:
            t += substring + str(index)
        t += self.tex_end
        return t

    def set_size(self,L):
        self.L = L

    def build_term_array(self):
        raise NotImplementedError()

class Sigmax(Fundamental):
    def __init__(self,index=0):
        Fundamental.__init__(self)
        self.index = index
        self.tex = [[r'\sigma_x^{',index]]
        self.tex_end = r'}'
        self.max_ind = index

    # TODO: allow symbolic index!
    def build_term_array(self):
        return np.array([(1<<self.index,0,self.coeff)],dtype=self.term_dtype())

class Sigmaz(Fundamental):
    def __init__(self,index=0):
        Fundamental.__init__(self)
        self.index = index
        self.tex = [[r'\sigma_z^{',index]]
        self.tex_end = r'}'
        self.max_ind = index

    def build_term_array(self):
        return np.array([(0,1<<self.index,self.coeff)],dtype=self.term_dtype())

class Sigmay(Fundamental):
    def __init__(self,index=0):
        Fundamental.__init__(self)
        self.index = index
        self.tex = [[r'\sigma_y^{',index]]
        self.tex_end = r'}'
        self.max_ind = index

    def build_term_array(self):
        return np.array([(1<<self.index,1<<self.index,1j*self.coeff)],dtype=self.term_dtype())
