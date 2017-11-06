
import unittest as ut

from itertools import product
from collections import OrderedDict
from tempfile import TemporaryFile,NamedTemporaryFile

dnm_args = []

from dynamite import config
config.initialize(dnm_args)
config.global_shell = False

import dynamite.operators as do
from dynamite.tools import build_state,vectonumpy
from dynamite._utils import coeff_to_str
from dynamite.extras import commutator, Majorana
import numpy as np
from petsc4py.PETSc import Sys,NormType
Print = Sys.Print

from helpers import *
import Hamiltonians

class SingleOperators(ut.TestCase):

    def setUp(self):
        self.L = 8

    def test_sigmax(self):
        for i in range(self.L):
            d,n = dnm_np_operator('x',index=i,L=self.L)
            r,msg = check_dnm_np(d,n)
            self.assertTrue(r,msg=msg)

    def test_sigmay(self):
        for i in range(self.L):
            d,n = dnm_np_operator('y',index=i,L=self.L)
            r,msg = check_dnm_np(d,n)
            self.assertTrue(r,msg=msg)

    def test_sigmaz(self):
        for i in range(self.L):
            d,n = dnm_np_operator('z',index=i,L=self.L)
            r,msg = check_dnm_np(d,n)
            self.assertTrue(r,msg=msg)

    def test_coeffs(self):
        for t in ['x','y','z']:
            with self.subTest(t=t):
                d,n = dnm_np_operator(t,index=1,L=self.L)
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_set_L(self):
        good_Ls = [2]
        for L in good_Ls:
            with self.subTest(L=L):
                s = do.Sigmax(1)
                s.L = L
                s.set_length(L)

        bad_Ls = [0,'hi',1]
        for L in bad_Ls:
            with self.subTest(L=L):
                with self.assertRaises(ValueError):
                    s = do.Sigmax(1)
                    s.L = L

    def test_dim(self):
        s = do.Identity()
        self.assertEqual(s.dim,None)

        s.L = self.L
        self.assertEqual(s.dim,2**self.L)

    def test_MSC_size(self):
        # simply an example in which I know the answer
        s = do.Sigmaz(0) + do.Sigmaz(1) + do.Sigmax(0)
        self.assertEqual(s.MSC_size,3)

    def test_shell_reset(self):
        s = do.Zero()
        s.L = 1

        s.build_mat()

        s.use_shell = False
        s.use_shell = True
        self.assertIs(s._mat,None)

        s.build_mat()
        s.use_shell = True
        self.assertIsNot(s._mat,None)

    def test_build(self):
        s = do.Sigmax()
        with self.assertRaises(ValueError):
            s.build_mat()
        s.L = 1

        # just make sure this doesn't throw an
        # exception
        s.get_mat(diag_entries=True)

    def test_zero(self):
        z = do.Zero(L=1)
        r,msg = check_dnm_np(z,np.array([[0.,0.],[0.,0.]],
                                        dtype=np.complex128))
        self.assertTrue(r,msg=msg)

    def test_identity(self):
        for L in range(1,5):
            with self.subTest(L=L):
                I = do.Identity(L=L)
                r,msg = check_dnm_np(I,
                                     np.identity(2**L))
                self.assertTrue(r,msg=msg)

class Products(ut.TestCase):

    def setUp(self):
        self.L = 4

    def test_scalar(self):
        for i in ['x','y','z']:
            with self.subTest(i=i):
                d,n = dnm_np_operator('x',index=0,L=1)
                r,msg = check_dnm_np(-2.1*d,-2.1*n)
                self.assertTrue(r,msg=msg)

                with self.assertRaises(TypeError):
                    [1,2] * d

    # NOTE: don't need to explicity check matrix-vector
    # multiply, because that is how we convert dynamite
    # matrix to numpy matrix during all checks

    def test_recursive(self):
        s = [do.Sigmax(i) for i in range(3)]

        s1 = s[0] * s[1] * s[2]
        s2 = s[0] * (s[1] * s[2])

        self.assertEqual(s1,s2)
        self.assertEqual(len(s1.terms),3)
        self.assertEqual(len(s2.terms),3)

        s1 = s[0] * s[1] * s[2] * s[0]
        s2 = (s[0] * s[1]) * (s[2] * s[0])
        self.assertEqual(s1,s2)
        self.assertEqual(len(s1.terms),4)
        self.assertEqual(len(s2.terms),4)

        with self.assertRaises(ValueError):
            do.Sum(terms=[])

    def test_alltwopoint(self):
        for i in range(self.L):
            for j in range(self.L):
                for t1,t2 in product(*[('x','y','z')]*2):
                    with self.subTest(i=i,j=j,sa=t1,sb=t2):
                        d1,n1 = dnm_np_operator(t1,
                                                index=i,
                                                L=self.L)
                        d2,n2 = dnm_np_operator(t2,
                                                index=j,
                                                L=self.L)
                        d = d1*d2
                        n = n1.dot(n2)

                        r,msg = check_dnm_np(d,n)
                        self.assertTrue(r,msg=msg)

    def test_IndexProduct_single(self):
        for s in ['x','y','z']:
            with self.subTest(s=s):
                d,n = dnm_np_operator(s,index=0,L=self.L)
                d = do.IndexProduct(d)
                for i in range(1,self.L):
                    n = n.dot(np_sigmai(s,index=i,L=self.L))
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_IndexProduct_limits(self):
        for low,high in [(1,1),(1,3),(0,3),(0,0)]:
            with self.subTest(low=low,high=high):
                d = do.Sigmay(L=self.L)
                d = do.IndexProduct(d,min_i=low,max_i=high)
                n = np_sigmay(low,self.L)
                for i in range(low+1,high+1):
                    n = n.dot(np_sigmay(i,self.L))
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_IndexProduct_edgelimits(self):
        for i in range(self.L):
            with self.subTest(ind=i):
                d = do.IndexProduct(do.Sigmax(L=self.L),
                                    min_i=i,max_i=i)
                r,msg = check_dnm_np(d,np_sigmax(i,self.L))
                self.assertTrue(r,msg=msg)

    def test_IndexProduct_exceptions(self):
        for low,high in [(1,0),(0,self.L-1),(0,self.L),(-1,0)]:
            with self.subTest(low=low,high=high):
                d = do.Sigmay(index=0) * do.Sigmay(index=1)
                d.L = self.L
                with self.assertRaises(ValueError):
                    do.IndexProduct(d,min_i=low,max_i=high)

    def test_length(self):
        x = do.Sigmax()
        y = do.Sigmay()
        x.L = 1
        y.L = 2
        with self.assertRaises(ValueError):
            x * y

class Sums(ut.TestCase):

    def setUp(self):
        self.L = 4

    def test_simple(self):
        d1,n1 = dnm_np_operator('y',index=0,L=self.L)
        d2,n2 = dnm_np_operator('z',index=1,L=self.L)
        d3,n3 = dnm_np_operator('x',index=3,L=self.L)

        r,msg = check_dnm_np(d1+d2+d3,n1+n2+n3)
        self.assertTrue(r,msg=msg)

    def test_sum_recursive(self):
        s = [do.Sigmax(i) for i in range(3)]

        s1 = s[0] + s[1] + s[2]
        s2 = s[0] + (s[1] + s[2])
        self.assertEqual(s1,s2)
        self.assertEqual(len(s1.terms),3)
        self.assertEqual(len(s2.terms),3)

        s1 = s[0] + s[1] + s[2] + s[0]
        s2 = (s[0] + s[1]) + (s[2] + s[0])
        self.assertEqual(s1,s2)
        self.assertEqual(len(s1.terms),4)
        self.assertEqual(len(s2.terms),4)

    def test_num_sum(self):
        H = do.Sigmax()
        with self.assertRaises(TypeError):
            H + 1

    def test_sum_one(self):
        d = do.Sum([do.Sigmax()])
        self.assertEqual(do.Sigmax(),d)

    def test_length(self):
        x = do.Sigmax()
        y = do.Sigmay()
        x.L = 1
        y.L = 2
        with self.assertRaises(ValueError):
            x + y

    def test_IndexSum(self):
        for s in ['x','y','z']:
            with self.subTest(s=s):
                d,n = dnm_np_operator(s,index=0,L=self.L)
                d = do.IndexSum(d)
                for i in range(1,self.L):
                    n = n + np_sigmai(s,index=i,L=self.L)
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_IndexSum_limits(self):
        for low,high in [(1,1),(1,3),(0,3),(0,0)]:
            with self.subTest(low=low,high=high):
                d = do.Sigmay(L=self.L)
                d = do.IndexSum(d,min_i=low,max_i=high)
                n = np_sigmay(low,self.L)
                for i in range(low+1,high+1):
                    n += np_sigmay(i,self.L)
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_IndexSum_edgelimits(self):
        for i in range(self.L):
            with self.subTest(ind=i):
                d = do.IndexSum(do.Sigmay(L=self.L),
                                min_i=i,max_i=i)
                n = np_sigmay(i,self.L)
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_IndexSum_exceptions(self):
        for low,high in [(1,0),(0,self.L+1),(-1,0)]:
            with self.subTest(low=low,high=high):
                d = do.Sigmay(L=self.L)
                with self.assertRaises(ValueError):
                    do.IndexSum(d,min_i=low,max_i=high)

    def test_IndexSum_wrap(self):
        d0,n0 = dnm_np_operator('x',L=self.L,index=0)
        d1,n1 = dnm_np_operator('x',L=self.L,index=1)

        # just to make sure this doesn't throw any exceptions
        d0.L = None
        d1.L = None
        d = do.IndexSum(d0*d1,wrap=True)
        d.L = self.L
        d.get_MSC()

        d0.L = self.L
        d1.L = self.L
        d = do.IndexSum(d0*d1,wrap=True)
        d.L = self.L

        n = n0.dot(n1)
        for i in range(1,self.L-1):
            s = np_sigmai('x',index=i,L=self.L)
            n += s.dot(np_sigmai('x',index=i+1,L=self.L))
        s = np_sigmai('x',index=self.L-1,L=self.L)
        n += s.dot(np_sigmai('x',index=0,L=self.L))

        r,msg = check_dnm_np(d,n)
        self.assertTrue(r,msg=msg)

class Compound(ut.TestCase):

    def setUp(self):
        config.global_L = 8
        self.op_lists = OrderedDict([
            ('XY',[(np_sigmax,np_sigmay),
                   (do.Sigmax,do.Sigmay)]),
            ('XYZ',[(np_sigmax,np_sigmay,np_sigmaz),
                    (do.Sigmax,do.Sigmay,do.Sigmaz)])
        ])

    def tearDown(self):
        config.global_L = None

    def test_Hamiltonians(self):
        for name in Hamiltonians.__all__:
            with self.subTest(name=name):
                d,n = getattr(Hamiltonians,name)(config.global_L)
                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_generatorSum(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.Sum(s() for s in ol[1])

                n = ol[0][0](0,config.global_L)
                for o in ol[0][1:]:
                    n += o(0,config.global_L)

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_generatorProd(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.Product(s() for s in ol[1])

                n = ol[0][0](0,config.global_L)
                for o in ol[0][1:]:
                    n = n.dot(o(0,config.global_L))

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_indexSumofSum(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.IndexSum(do.Sum(s() for s in ol[1]))

                n = ol[0][0](0,config.global_L)
                for o in ol[0][1:]:
                    n += o(0,config.global_L)

                for i in range(1,config.global_L):
                    for o in ol[0]:
                        n += o(i,config.global_L)

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_indexSumofProd(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.IndexSum(do.Product(s() for s in ol[1]))

                n = ol[0][0](0,config.global_L)
                for o in ol[0][1:]:
                    n = n.dot(o(0,config.global_L))

                for i in range(1,config.global_L):
                    n_tmp = ol[0][0](i,config.global_L)
                    for o in ol[0][1:]:
                        n_tmp = n_tmp.dot(o(i,config.global_L))
                    n += n_tmp

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_indexProdofSum(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.IndexProduct(do.Sum(s() for s in ol[1]))

                n = ol[0][0](0,config.global_L)
                for o in ol[0][1:]:
                    n += o(0,config.global_L)

                for i in range(1,config.global_L):
                    n_tmp = ol[0][0](i,config.global_L)
                    for o in ol[0][1:]:
                        n_tmp += o(i,config.global_L)
                    n = n.dot(n_tmp)

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_indexProdofProd(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.IndexProduct(do.Product(s() for s in ol[1]))

                n = ol[0][0](0,config.global_L)
                for o in ol[0][1:]:
                    n = n.dot(o(0,config.global_L))

                for i in range(1,config.global_L):
                    n_tmp = ol[0][0](i,config.global_L)
                    for o in ol[0][1:]:
                        n_tmp = n_tmp.dot(o(i,config.global_L))
                    n = n.dot(n_tmp)

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_SumofProduct(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.Sum(s(0)*s(1) for s in ol[1])

                s = ol[0][0](0,config.global_L)
                n = s.dot(ol[0][0](1,config.global_L))
                for o in ol[0][1:]:
                    s = o(0,config.global_L)
                    n += s.dot(o(1,config.global_L))

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

    def test_indexSumofSumofProduct(self):
        for name,ol in self.op_lists.items():
            with self.subTest(ops=name):
                d = do.IndexSum(do.Sum(s(0)*s(1)
                                       for s in ol[1]))

                s = ol[0][0](0,config.global_L)
                n = s.dot(ol[0][0](1,config.global_L))
                for o in ol[0][1:]:
                    s = o(0,config.global_L)
                    n += s.dot(o(1,config.global_L))

                for i in range(1,config.global_L-1):
                    for o in ol[0]:
                        s = o(i,config.global_L)
                        n += s.dot(o(i+1,config.global_L))

                r,msg = check_dnm_np(d,n)
                self.assertTrue(r,msg=msg)

class StateBuilding(ut.TestCase):

    def setUp(self):
        config.global_L = 4
        self.L = config.global_L

    def tearDown(self):
        config.global_L = None

    def test_buildstate(self):
        for i in [0,
                  # some randomly-picked states
                  int(0.339054706405*(2**self.L)),
                  int(0.933703666179*(2**self.L))]:
            with self.subTest(state=i):
                d = build_state(state=i)
                n = np.zeros(2**self.L,dtype=np.complex128)
                n[i] = 1

                r,msg = check_vecs(d,n)
                self.assertTrue(r,msg=msg)

    def test_str_buildstate(self):
        for i in ['UUUU','UDUD','UDDD']:
            with self.subTest(state=i):
                d = build_state(state=i)
                n = np.zeros(2**self.L,dtype=np.complex128)
                ind = int(i.replace('U','1').replace('D','0'),2)
                n[ind] = 1

                r,msg = check_vecs(d,n)
                self.assertTrue(r,msg=msg)

    def test_random(self):
        # just make sure that we can build random states
        # without exceptions and that their norm is reasonable.
        # it would take too long to actually test that they
        # are correctly distributed etc during the automated tests.
        s = build_state(L=self.L,state='random')
        r,msg = check_close(s.norm(),1)
        self.assertTrue(r,msg=msg)

        s = build_state(L=self.L,state='random',seed=0)
        x = np.random.random()
        t = build_state(L=self.L,state='random',seed=0)
        y = np.random.random()

        r,msg = check_close(s.norm(),1)
        self.assertTrue(r,msg=msg)

        r,msg = check_close(s.dot(t),1)
        self.assertTrue(r,msg=msg)

        # make sure we aren't screwing with numpy's random
        # number generator in a way users wouldn't expect
        self.assertNotEqual(x,y)

    def test_buildstate_exceptions(self):
        for i in ['U','UDDUDUD','DUDE',10000,-1]:
            with self.subTest(state=i):
                with self.assertRaises(ValueError):
                    build_state(state=i)

        for i in [1j,4.2]:
            with self.subTest(state=i):
                with self.assertRaises(TypeError):
                    build_state(state=i)

class Evolve(ut.TestCase):

    def setUp(self):
        self.L = 6
        self.test_states = [0,
                            int(0.339054706405*(2**self.L)),
                            int(0.933703666179*(2**self.L)),
                            'random']

    def test_Hamiltonians(self):
        for name in Hamiltonians.__all__:
            for state in self.test_states:
                with self.subTest(H=name,state=state):
                    d,n = getattr(Hamiltonians,name)(self.L)
                    r,msg = check_evolve(d,n,state)
                    self.assertTrue(r,msg=msg)

class Eigsolve(ut.TestCase):

    def setUp(self):
        self.L = 6

    def test_Hamiltonians(self):
        for name in Hamiltonians.__all__:
            with self.subTest(H=name):
                d,n = getattr(Hamiltonians,name)(self.L)
                with self.subTest(which='smallest'):
                    self.check_eigs(d,n)

                with self.subTest(which='smallest_4'):
                    self.check_eigs(d,n,nev=4)

                with self.subTest(which='exterior'):
                    self.check_eigs(d,n,which='exterior')

                # these tests don't work for the XXYY Hamiltonian for
                # mathematical reasons (the matrix has singular
                # rows so you can't do a normal linear solve)

                # there seems to be some way to deal with this in
                # SLEPc with MatSetNullSpace

                # they also will fail in parallel if a parallel linear
                # solver was not included in the PETSc build (e.g. with
                # --download-mumps option to ./configure)
                # TODO: check if package exists, if not don't run these tests

                if not config.global_shell:
                    if name != 'XXYY':
                        with self.subTest(which='target0'):
                            self.check_eigs(d,n,target=0,nev=2)

                        with self.subTest(which='target-1.2'):
                            self.check_eigs(d,n,target=-1.2,nev=2)

                        with self.subTest(which='targetexcept'):
                            with self.assertRaises(ValueError):
                                self.check_eigs(d,n,which='target')
                else:
                    with self.subTest(which='target'):
                        with self.assertRaises(TypeError):
                            self.check_eigs(d,n,target=0)


    def check_eigs(self,d,n,**kwargs):

        evs,evecs = d.eigsolve(getvecs=True,**kwargs)
        nevs,_ = np.linalg.eigh(n)

        if 'nev' in kwargs:
            self.assertGreater(len(evs),kwargs['nev']-1)
        else:
            self.assertGreater(len(evs),0)

        # TODO: should check 'target' option actually gives eigs
        # closest to target

        # make sure every eigenvalue is close to one in the list
        # also check that the eigenvector is correct
        for ev,evec in zip(evs,evecs):
            with self.subTest(ev=ev):
                # there are some matching eigenvalues
                if not np.abs(nevs-ev).min() < 1E-12:
                    pass
                    #print(evs,nevs,n)
                self.assertLess(np.abs(nevs-ev).min(),1E-12)

                # check that the eigenvector is
                # a) an eigenvector and
                # b) has the right eigenvalue
                if ev != 0:
                    err = d.get_mat()*evec / ev - evec
                else:
                    err = d.get_mat()*evec
                errnorm = err.norm(NormType.INFINITY)
                vecnorm = evec.norm(NormType.INFINITY)
                self.assertLess(errnorm,1E-6*vecnorm)

from dynamite.computations import reduced_density_matrix
from dynamite.computations import entanglement_entropy
from petsc4py.PETSc import Vec

# this test uses qutip to test the entanglement entropy computation
# if we don't have qutip, just skip it

try:
    import qutip as qtp
except ImportError:
    qtp = None

@ut.skipIf(qtp is None,'could not find QuTiP')
class Entropy(ut.TestCase):

    def setUp(self):
        self.L = 4
        self.cuts = [0,1,2,4]
        self.states = OrderedDict([
            ('product0',build_state(L=self.L)),
            ('product1',
             build_state(L=self.L,
                         state=int(0.8675309*(2**self.L)))),
            ('random',build_state(L=self.L,state='random'))
        ])

        H = do.IndexSum(do.Sum(s(0)*s(1)
                               for s in (do.Sigmax,do.Sigmaz)))
        H.L = self.L
        self.states['evolved'] = H.evolve(self.states['product1'],
                                          1.0)

    def test_dm_entropy(self):
        for cut in self.cuts:
            for name,state in self.states.items():
                with self.subTest(cut=cut,state=name):
                    ddm = reduced_density_matrix(state,cut)
                    dy_EE = entanglement_entropy(state,cut)

                    qtp_state = qtp.Qobj(vectonumpy(state),
                                         dims=[[2]*self.L,
                                               [1]*self.L])

                    dm = qtp_state * qtp_state.dag()

                    if cut > 0:
                        dm = dm.ptrace(list(range(cut)))
                    else:
                        # qutip breaks when you ask it to trace out everything
                        # maybe I should submit a pull request to them
                        dm = None

                    if dm is not None:
                        r,msg = check_allclose(dm.full(),ddm)
                        self.assertTrue(r,msg=msg)
                        qtp_EE = qtp.entropy_vn(dm)
                    else:
                        r,msg = check_allclose(ddm,np.array([[1.+0.0j]]))
                        self.assertTrue(r,msg=msg)
                        qtp_EE = 0

                    r,msg = check_close(qtp_EE,dy_EE)
                    self.assertTrue(r,msg=msg)

class Save(ut.TestCase):
    def test_SaveAndLoad(self):
        H,n = Hamiltonians.longrange(10)

        # test saving and loading by both file name
        # and file object
        files = [('by_file',TemporaryFile),
                 ('by_name','save_test.msc')]

        for t,fg in files:
            with self.subTest(ftype=t):

                if isinstance(fg,str):
                    f = fg
                else:
                    f = fg()

                with self.subTest(m='save'):
                    H.save(f)

                if not isinstance(f,str):
                    f.seek(0)

                with self.subTest(m='load'):
                    Hf = do.Load(f)
                    self.assertEqual(H,Hf)

                    # also make sure that the generated PETSc matrix
                    # works
                    r,msg = check_dnm_np(Hf,n)
                    self.assertTrue(r,msg=msg)

                if not isinstance(fg,str):
                    f.close()

class Utils(ut.TestCase):

    def setUp(self):

        # formatted as (x,signs,expected string)
        self.cases = [
            (-1,'+-','-'),
            (1,'+-','+'),
            (1,'-',''),
            (0,'-','0'),
            (-0.1,'-','-0.1'),
            (-0.1111111,'-','-0.111'),
            (-0.1111111,'','0.111'),
            (1.1111111,'+-','+1.11')
        ]

    def test_coefftostr(self):

        for case in self.cases:
            with self.subTest(x=case[0],signs=case[1]):
                self.assertEqual(coeff_to_str(case[0],
                                              signs=case[1]),case[2])

class Extras(ut.TestCase):

    def test_Majorana(self):
        tests = [
            (1,do.Sigmay()),
            (2,do.Sigmaz(0)*do.Sigmax(1)),
            (4,do.Sigmaz(0)*do.Sigmaz(1)*do.Sigmax(2))
        ]

        for idx,op in tests:
            X = Majorana(idx)
            self.assertEqual(X,op)

    def test_commutator(self):
        self.assertEqual(commutator(do.Sigmax(),do.Sigmay()),2j*do.Sigmaz())

from dynamite.tools import track_memory,get_max_memory_usage,get_cur_memory_usage
class Benchmarking(ut.TestCase):

    # just test that there are no exceptions thrown
    # don't think we need to make sure the values returned
    # are actually correct, that's not our problem

    def test_memtracking(self):
        get_cur_memory_usage()

        # this isn't implemented yet
        # TODO: should raise RuntimeError if get_max_memory_usage
        # is called without a preceding call to track_memory
        # with self.assertRaises(RuntimeError):
        #     get_max_memory_usage()

        track_memory()
        get_max_memory_usage()

class Config(ut.TestCase):

    def test_global_L(self):

        config.global_L = 10

        test_ops = OrderedDict([
            ('sx', do.Sigmax),
            ('sy', do.Sigmay),
            ('sz', do.Sigmaz),
            ('ident', do.Identity),
            ('zero', do.Zero),
            ('sum', lambda: do.Sum([do.Sigmax()])),
            ('product', lambda: do.Product([do.Sigmax()])),
            ('indexsum', lambda: do.IndexSum(do.Sigmax())),
            ('indexproduct', lambda: do.IndexProduct(do.Sigmax())),
        ])

        for op,d in test_ops.items():
            with self.subTest(op=op):
                self.assertEqual(d().L,10)

        v = build_state()
        self.assertEqual(v.size,2**10)

        config.global_L = None

        for op,d in test_ops.items():
            with self.subTest(op=op):
                self.assertIs(d().L,None)

        with self.assertRaises(ValueError):
            build_state()

if __name__ == '__main__':

    # only get output from one process
    # from sys import stderr
    # from os import devnull
    # if PROC_0:
    #     stream = stderr
    # else:
    #     stream = open(devnull,'w')
    # ut.main(testRunner=ut.TextTestRunner(stream=stream))

    ut.main()
