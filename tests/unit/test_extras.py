'''
Unit tests for extras.py.

These tests should NOT require MPI.
'''

import unittest as ut
import numpy as np

from dynamite import extras
from dynamite.operators import sigmax, sigmay, sigmaz

class Commutator(ut.TestCase):

    def test_pauli_relations(self):

        pauli_orders = [[sigmax(0), sigmay(0), sigmaz(0)],
                        [sigmay(0), sigmaz(0), sigmax(0)],
                        [sigmaz(0), sigmax(0), sigmay(0)]]

        for op1, op2, op3 in pauli_orders:         
            comm = extras.commutator(op1, op2)
            self.assertEqual(comm, 2j*op3)

#class Majorana(ut.TestCase):
#    def test_majoranas(self):

if __name__ == '__main__':
    ut.main()