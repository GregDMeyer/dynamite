
import unittest as ut
import dynamite as dy
import numpy as np
import qutip as qtp
from itertools import product
# from petsc4py.PETSc import Scatter

def to_np(H):
    ret = np.ndarray((1<<H.L,1<<H.L),dtype=np.complex128)
    s1 = dy.build_state(H.L)
    s2 = dy.build_state(H.L)
#    scat,s0 = Scatter.toZero(s2) # need to scatter for >1 process
    for i in range(1<<H.L):
        s1.set(0)
        s1.setValue(i,1)
        s1.assemblyBegin()
        s1.assemblyEnd()
        s2 = H*s1
        ret[:,i] = s2[:]

    return ret

def identity_product(op, index, L):
    ret = None
    for i in range(L):
        if i == index:
            this_op = op
        else:
            this_op = qtp.identity(2)
        if ret is None:
            ret = this_op
        else:
            ret = qtp.tensor(this_op,ret)
    return ret

def q_sigmax(index=0,L=1):
    return identity_product(qtp.sigmax(),index,L)

def q_sigmay(index=0,L=1):
    return identity_product(qtp.sigmay(),index,L)

def q_sigmaz(index=0,L=1):
    return identity_product(qtp.sigmaz(),index,L)

def q_sigmai(t,*args,**kwargs):
    if t == 'sx':
        return q_sigmax(*args,**kwargs)
    elif t == 'sy':
        return q_sigmay(*args,**kwargs)
    elif t == 'sz':
        return q_sigmaz(*args,**kwargs)
    else:
        raise Exception('Type \'%s\' is not valid.' % t)

def get_both(op_type,index=0,L=1):
    if op_type == 'sx':
        sx = dy.Sigmax(index)
        sx.set_size(L)
        return (sx,q_sigmax(index,L))
    elif op_type == 'sy':
        sy = dy.Sigmay(index)
        sy.set_size(L)
        return (sy,q_sigmay(index,L))
    elif op_type == 'sz':
        sz = dy.Sigmaz(index)
        sz.set_size(L)
        return (sz,q_sigmaz(index,L))

class SingleOperators(ut.TestCase):

    def setUp(self):
        self.L = 4

    def test_sigmax(self):
        for i in range(self.L):
            dsx,qsx = get_both('sx',index=i,L=self.L)
            self.assertTrue(np.all(to_np(dsx)==qsx.full()))

    def test_sigmay(self):
        for i in range(self.L):
            dsy,qsy = get_both('sy',index=i,L=self.L)
            self.assertTrue(np.all(to_np(dsy)==qsy.full()))

    def test_sigmaz(self):
        for i in range(self.L):
            dsz,qsz = get_both('sz',index=i,L=self.L)
            self.assertTrue(np.all(to_np(dsz)==qsz.full()))

    # TODO: check identity, etc

class Products(ut.TestCase):

    def setUp(self):
        self.L = 4

    def check_product(self,t1,i1,t2,i2):
        ds1,qs1 = get_both(t1,index=i1,L=self.L)
        ds2,qs2 = get_both(t2,index=i2,L=self.L)
        self.assertTrue(np.all(to_np(ds1*ds2)==(qs1*qs2).full()))

    def test_alltwopoint(self):
        for i in range(self.L):
            for j in range(self.L):
                for t1,t2 in product(*[('sx','sy','sz')]*2):
                    with self.subTest(i=i,j=j,sa=t1,sb=t2):
                        self.check_product(t1,i,t2,j)

    def test_PiProd_single(self):
        for s in ['sx','sy','sz']:
            with self.subTest(s=s):
                ds,qs = get_both(s,L=self.L)
                p = dy.PiProd(ds)
                for i in range(1,self.L):
                    qs = qs * q_sigmai(s,index=i,L=self.L)
                self.assertTrue(np.all(to_np(p)==qs.full()))

if __name__ == '__main__':
    ut.main()
