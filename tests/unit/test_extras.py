
import unittest as ut

from dynamite.operators import sigmax, sigmay, sigmaz, identity, zero
from dynamite.extras import commutator, majorana


class Commutator(ut.TestCase):

    def test_paulis(self):
        self.assertEqual(commutator(sigmax(), sigmay()), 2j*sigmaz())
        self.assertEqual(commutator(sigmaz(), sigmay()), -2j*sigmax())

    def test_str(self):
        self.assertEqual(str(commutator(sigmax(1), sigmay(2))), '[σx[1], σy[2]]')

    def test_tex(self):
        self.assertEqual(commutator(sigmax(1), sigmay(2))._repr_latex_(), '$\\left[ \\sigma^x_{1}, \\sigma^y_{2} \\right]$')

    def test_repr(self):
        expr = 'commutator(sigmax(1), sigmay(2))'
        self.assertEqual(repr(eval(expr)), expr)


class Majorana(ut.TestCase):

    def test_relations(self):
        # majorana is its own antiparticle
        self.assertEqual(majorana(0)*majorana(0), identity())
        self.assertEqual(majorana(10)*majorana(10), identity())

        # majoranas anticommute with each other
        self.assertEqual(majorana(0)*majorana(1) + majorana(1)*majorana(0), zero())
        self.assertEqual(majorana(0)*majorana(2) + majorana(2)*majorana(0), zero())
        self.assertEqual(majorana(10)*majorana(1) + majorana(1)*majorana(10), zero())

    def test_str(self):
        self.assertEqual(str(majorana(0)), 'χ[0]')
        self.assertEqual(str(majorana(1)), 'χ[1]')

    def test_tex(self):
        self.assertEqual(majorana(0)._repr_latex_(), '$\\chi_{0}$')
        self.assertEqual(majorana(1)._repr_latex_(), '$\\chi_{1}$')

    def test_repr(self):
        expr = 'majorana(1)'
        self.assertEqual(repr(eval(expr)), expr)


if __name__ == '__main__':
    ut.main()
