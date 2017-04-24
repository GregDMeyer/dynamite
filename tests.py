
from itertools import product

import unittest as ut
import dynamite as dy
import numpy as np
import qutip as qtp
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

class BaseTest(ut.TestCase):

    def check_dy_qtp(self,dy_obj,qtp_obj):
        self.assertTrue(np.all(to_np(dy_obj)==qtp_obj.full()))
        self.assertEqual(dy_obj.build_qutip(),qtp_obj)

class SingleOperators(BaseTest):

    def setUp(self):
        self.L = 4

    def test_sigmax(self):
        for i in range(self.L):
            dsx,qsx = get_both('sx',index=i,L=self.L)
            self.check_dy_qtp(dsx,qsx)

    def test_sigmay(self):
        for i in range(self.L):
            dsy,qsy = get_both('sy',index=i,L=self.L)
            self.check_dy_qtp(dsy,qsy)

    def test_sigmaz(self):
        for i in range(self.L):
            dsz,qsz = get_both('sz',index=i,L=self.L)
            self.check_dy_qtp(dsz,qsz)

    def test_coeffs(self):
        for t in ['sx','sy','sz']:
            with self.subTest(t=t):
                ds,qs = get_both(t,index=1,L=self.L)
                self.check_dy_qtp(-0.5*ds,-0.5*qs)

    def test_zero(self):
        self.assertTrue(np.all(to_np(dy.Zero(L=1))==np.array([[0.,0.],[0.,0.]])))
        self.assertEqual(dy.Zero(L=1).build_qutip(),qtp.Qobj([[0,0],[0,0]]))

    def test_identity(self):
        for L in range(1,5):
            with self.subTest(L=L):
                self.assertTrue(np.all(to_np(dy.Identity(L=L))==np.identity(1<<L)))

                ident = qtp.identity(2)
                for _ in range(1,L):
                    ident = qtp.tensor(ident,qtp.identity(2))
                self.assertEqual(dy.Identity(L=L).build_qutip(),ident)

class Products(BaseTest):

    def setUp(self):
        self.L = 4

    def check_product(self,t1,i1,t2,i2):
        ds1,qs1 = get_both(t1,index=i1,L=self.L)
        ds2,qs2 = get_both(t2,index=i2,L=self.L)
        d = ds1*ds2
        q = qs1*qs2
        self.check_dy_qtp(d,q)

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
                self.check_dy_qtp(p,qs)

    def test_PiProd_limits(self):
        for low,high in [(1,1),(1,3),(0,3),(0,0)]:
            with self.subTest(low=low,high=high):
                dsy,_ = get_both('sy',L=self.L)
                prod = dy.PiProd(dsy,min_i=low,max_i=high)
                q = q_sigmay(low,self.L)
                for i in range(low+1,high+1):
                    q *= q_sigmay(i,self.L)
                self.check_dy_qtp(prod,q)

    def test_PiProd_edgelimits(self):
        for i in range(self.L):
            with self.subTest(ind=i):
                prod = dy.PiProd(dy.Sigmax(L=self.L),min_i=i,max_i=i)
                self.check_dy_qtp(prod,q_sigmax(i,self.L))

    def test_PiProd_exceptions(self):
        for low,high in [(1,0),(0,self.L+1),(-1,0)]:
            with self.subTest(low=low,high=high):
                dsy,_ = get_both('sy',L=self.L)
                with self.assertRaises(IndexError):
                    dy.PiProd(dsy,min_i=low,max_i=high)

class Sums(BaseTest):

    def setUp(self):
        self.L = 4

    def test_SumTerms(self):
        ds1,qs1 = get_both('sy',index=0,L=self.L)
        ds2,qs2 = get_both('sz',index=1,L=self.L)
        ds3,qs3 = get_both('sx',index=3,L=self.L)
        self.check_dy_qtp(ds1+ds2+ds3,qs1+qs2+qs3)

    def test_SigmaSum(self):
        for s in ['sx','sy','sz']:
            with self.subTest(s=s):
                ds,qs = get_both(s,L=self.L)
                p = dy.SigmaSum(ds)
                for i in range(1,self.L):
                    qs = qs + q_sigmai(s,index=i,L=self.L)
                self.check_dy_qtp(p,qs)

    def test_SigmaSum_limits(self):
        for low,high in [(1,1),(1,3),(0,3),(0,0)]:
            with self.subTest(low=low,high=high):
                dsy,_ = get_both('sy',L=self.L)
                s = dy.SigmaSum(dsy,min_i=low,max_i=high)
                q = q_sigmay(low,self.L)
                for i in range(low+1,high+1):
                    q += q_sigmay(i,self.L)
                self.check_dy_qtp(s,q)

    def test_SigmaSum_edgelimits(self):
        for i in range(self.L):
            with self.subTest(ind=i):
                s = dy.SigmaSum(dy.Sigmay(L=self.L),min_i=i,max_i=i)
                self.check_dy_qtp(s,q_sigmay(i,self.L))

    def test_SigmaSum_exceptions(self):
        for low,high in [(1,0),(0,self.L+1),(-1,0)]:
            with self.subTest(low=low,high=high):
                dsy,_ = get_both('sy',L=self.L)
                with self.assertRaises(IndexError):
                    dy.SigmaSum(dsy,min_i=low,max_i=high)

class Compound(BaseTest):

    def setUp(self):
        self.L = 8
        self.op_lists = {
            'XY':[(q_sigmax,q_sigmay),(dy.Sigmax,dy.Sigmay)],
            'XYZ':[(q_sigmax,q_sigmay,q_sigmaz),(dy.Sigmax,dy.Sigmay,dy.Sigmaz)]
        }

    def test_Ising(self):
        H = dy.SigmaSum(dy.Sigmaz()*dy.Sigmaz(1)) + 0.5*dy.SigmaSum(dy.Sigmax())
        H.set_size(self.L)

        qz = q_sigmaz(index=0,L=self.L) * q_sigmaz(index=1,L=self.L)
        for i in range(1,self.L-1):
            qz += q_sigmaz(index=i,L=self.L) * q_sigmaz(index=i+1,L=self.L)

        qx = q_sigmax(index=0,L=self.L)
        for i in range(1,self.L):
            qx += q_sigmax(index=i,L=self.L)

        qH = qz + 0.5*qx

        self.check_dy_qtp(H,qH)

    def test_generatorSum(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):

                H = dy.SumTerms(s() for s in ol[1])
                H.set_size(self.L)

                qH = ol[0][0](0,self.L)
                for o in ol[0][1:]:
                    qH += o(0,self.L)

                self.check_dy_qtp(H,qH)

    def test_generatorProd(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.Product(s() for s in ol[1])
                H.set_size(self.L)

                qH = ol[0][0](0,self.L)
                for o in ol[0][1:]:
                    qH *= o(0,self.L)

                self.check_dy_qtp(H,qH)

    def test_indexSumofSum(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.SigmaSum(dy.SumTerms(s() for s in ol[1]))
                H.set_size(self.L)

                qH = ol[0][0](0,self.L)
                for o in ol[0][1:]:
                    qH += o(0,self.L)

                for i in range(1,self.L):
                    for o in ol[0]:
                        qH += o(i,self.L)

                self.check_dy_qtp(H,qH)

    def test_indexSumofProd(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.SigmaSum(dy.Product(s() for s in ol[1]))
                H.set_size(self.L)

                qH = ol[0][0](0,self.L)
                for o in ol[0][1:]:
                    qH *= o(0,self.L)

                for i in range(1,self.L):
                    qH_tmp = ol[0][0](i,self.L)
                    for o in ol[0][1:]:
                        qH_tmp *= o(i,self.L)
                    qH += qH_tmp

                self.check_dy_qtp(H,qH)

    def test_indexProdofSum(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.PiProd(dy.SumTerms(s() for s in ol[1]))
                H.set_size(self.L)

                qH = ol[0][0](0,self.L)
                for o in ol[0][1:]:
                    qH += o(0,self.L)

                for i in range(1,self.L):
                    qH_tmp = ol[0][0](i,self.L)
                    for o in ol[0][1:]:
                        qH_tmp += o(i,self.L)
                    qH *= qH_tmp

                self.check_dy_qtp(H,qH)

    def test_indexProdofProd(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.PiProd(dy.Product(s() for s in ol[1]))
                H.set_size(self.L)

                qH = ol[0][0](0,self.L)
                for o in ol[0][1:]:
                    qH *= o(0,self.L)

                for i in range(1,self.L):
                    qH_tmp = ol[0][0](i,self.L)
                    for o in ol[0][1:]:
                        qH_tmp *= o(i,self.L)
                    qH *= qH_tmp

                self.check_dy_qtp(H,qH)

    def test_SumofProduct(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.SumTerms(s(0)*s(1) for s in ol[1])
                H.set_size(self.L)

                qH = ol[0][0](0,self.L) * ol[0][0](1,self.L)
                for o in ol[0][1:]:
                    qH += o(0,self.L) * o(1,self.L)

                self.check_dy_qtp(H,qH)

    def test_indexSumofSumofProduct(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                H = dy.SigmaSum(dy.SumTerms(s(0)*s(1) for s in ol[1]))
                H.set_size(self.L)

                qH = ol[0][0](0,self.L) * ol[0][0](1,self.L)
                for o in ol[0][1:]:
                    qH += o(0,self.L) * o(1,self.L)

                for i in range(1,self.L-1):
                    for o in ol[0]:
                        qH += o(i,self.L) * o(i+1,self.L)

                self.check_dy_qtp(H,qH)

class StateBuilding(BaseTest):

    def setUp(self):
        self.L = 4

    def test_buildstate(self):
        for i in [0,int(0.79737*(1<<self.L))]: # some random state I picked
            with self.subTest(init_state=i):
                s = dy.build_state(L=self.L,init_state=i)
                qs = qtp.basis(2,i&1)
                for j in range(1,self.L):
                    qs = qtp.tensor(qtp.basis(2,(i>>j)&1),qs)
                self.assertTrue(np.all(s[:]==qs.full().flatten()))

class Evolution(BaseTest):

    def setUp(self):
        self.L = 6
        self.test_states = [0,int(0.79737*(1<<self.L))]

    def check_solve(self,dH,init_state,t,tol=1E-7):
        ds = dy.build_state(L=self.L,init_state=init_state)
        r = dH.evolve(ds,t=t)

        qH = dH.build_qutip()
        qs = qtp.basis(2,init_state&1)
        for j in range(1,self.L):
            qs = qtp.tensor(qtp.basis(2,(init_state>>j)&1),qs)
        qres = qtp.sesolve(qH,qs,[0,t])
        qr = qres.states[1].full().flatten()
        res = r[:]
        self.assertGreater(np.abs(res.conj().dot(qr)),1-tol)

    def test_Identity(self):
        for i in self.test_states: # some random state I picked
            with self.subTest(init_state=i):
                H = dy.Identity(L=self.L)
                self.check_solve(H,i,1.0)

    def test_ising(self):
        for i in self.test_states:
            with self.subTest(init_state=i):
                H = dy.SigmaSum(dy.Sigmaz()*dy.Sigmaz(1)) + 0.5*dy.SigmaSum(dy.Sigmax())
                H.set_size(self.L)
                self.check_solve(H,i,1.0)

    def test_XXYY(self):
        for i in self.test_states:
            with self.subTest(init_state=i):
                H = dy.SigmaSum(dy.SumTerms(s(0)*s(1) for s in [dy.Sigmax,dy.Sigmay]))
                H.set_size(self.L)
                self.check_solve(H,i,1.0)

if __name__ == '__main__':
    ut.main()
