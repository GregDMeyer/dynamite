
import unittest as ut
import numpy as np
from dynamite import config
from dynamite.operators import Sigmax,Sigmay,Sigmaz
from dynamite.subspace import Parity
from dynamite.tools import build_state,vectonumpy

config.L = 12
config.shell = True

class Mapping(ut.TestCase):

    def setUp(self):
        self.L = config.L
        self.even = Parity('even',self.L)
        self.odd = Parity('odd',self.L)

    def test_forward_back(self):
        idxs = np.arange(1<<(self.L-1))

        with self.subTest(parity="even"):
            s = self.even.state_to_idx(self.even.idx_to_state(idxs))
            self.assertTrue(np.all(s==idxs))

        with self.subTest(parity="odd"):
            s = self.odd.state_to_idx(self.odd.idx_to_state(idxs))
            self.assertTrue(np.all(s==idxs))

    # just in case one wants to see them
    def print_idx_to_state(self):
        idxs = np.arange(1<<(self.L-1))
        print('even:\n')
        s = self.even.idx_to_state(idxs)
        for i in range(1<<(self.L-1)):
            print('',i,bin(s[i])[2:].zfill(self.L))

        print('\nodd:\n')
        s = self.odd.idx_to_state(idxs)
        for i in range(1<<(self.L-1)):
            print('',i,bin(s[i])[2:].zfill(self.L))

    def print_state_to_idx(self):
        states = np.arange(1<<(self.L))
        print('even:\n')
        idxs = self.even.state_to_idx(states)
        for i in range(1<<(self.L)):
            print('',i,bin(idxs[i])[2:].zfill(self.L-1) if not idxs[i]==-1 else -1)

        print('\nodd:\n')
        idxs = self.odd.state_to_idx(states)
        for i in range(1<<(self.L)):
            print('',i,bin(idxs[i])[2:].zfill(self.L-1) if not idxs[i]==-1 else -1)

class PtoP(ut.TestCase):

    def setUp(self):
        self.s = build_state(state='random')

    def test_same(self):
        # this Hamiltonian conserves parity
        H = Sigmaz() + 0.1*Sigmax(0)*Sigmay(2) + 0.3*Sigmay(1)*Sigmay(2) \
                + 0.7*Sigmay(0)*Sigmax(1)*Sigmax(2)*Sigmay(3) + Sigmaz(config.L-1)
        r1 = H*self.s

        for p in (0,1):
            with self.subTest(parity=p):
                sp = convert_state(self.s,p)
                H.subspace = Parity(p,L=config.L)
                r2 = H*sp

                # if not np.abs(r2.norm()**2-convert_state(r1,p).dot(r2)) < 1E-15:
                #     convert_state(r1,p).view()
                #     r2.view()
                self.assertLess(np.abs(r2.norm()**2-convert_state(r1,p).dot(r2)),1E-15)

    def test_anti(self):
        # this one anti-conserves parity
        H = Sigmay() + 0.1*Sigmaz(0)*Sigmay(2) + 0.3*Sigmax(1)*Sigmaz(2) \
                + 0.7*Sigmay(0)*Sigmax(1)*Sigmaz(2)*Sigmay(3) + Sigmay(config.L-1)
        r1 = H*self.s

        for pr,pl in ((0,1),(1,0)):
            with self.subTest(pr=pr,pl=pl):
                sp = convert_state(self.s,pr)
                H.right_subspace = Parity(pr,L=config.L)
                H.left_subspace = Parity(pl,L=config.L)
                r2 = H*sp

                self.assertLess(np.abs(r2.norm()**2-convert_state(r1,pl).dot(r2)),1E-15)

def convert_state(s,p):
    sn = vectonumpy(s,toall=True)
    P = Parity(p,config.L)

    sstart,send = s.getOwnershipRange()

    r = build_state(L=config.L-1)
    idxs = P.state_to_idx(np.arange(sstart,send))
    r.setValues(idxs[idxs!=-1].astype(np.int32),sn[np.arange(sstart,send)[idxs!=-1]])
    r.assemblyBegin()
    r.assemblyEnd()
    return r

if __name__ == "__main__":
    ut.main()
