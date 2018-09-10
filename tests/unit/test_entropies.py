'''
Test computation of Von Neumann and Renyi entropies.
'''

import unittest as ut
import numpy as np

from dynamite.computations import dm_entanglement_entropy, dm_renyi_entropy

class VonNeumann(ut.TestCase):

    def check_entropy(self, dm, correct):
        check = dm_entanglement_entropy(dm)
        eps = 1E-14
        self.assertTrue(np.isclose(check, correct, rtol=0, atol=eps))

    def test_pure_state(self):
        dm = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        self.check_entropy(dm, 0.0)

    def test_pure_X_state(self):
        dm = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        self.check_entropy(dm, 0.0)

    def test_pure_Y_state(self):
        dm = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
        self.check_entropy(dm, 0.0)

    def test_diag_mixed(self):
        dm = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
        self.check_entropy(dm, np.log(2))

    def test_diag_mixed2(self):
        dm = np.array([[3/4, 0], [0, 1/4]], dtype=np.complex128)
        self.check_entropy(dm, -(3/4)*np.log(3)+np.log(4))

    def test_random(self):
        dm = np.array([
            [ (0.51401+0j), (-0.022913+0.007162j) ],
            [ (-0.022913-0.007162j), (0.485675+0j) ],
        ], dtype=np.complex128)
        self.check_entropy(dm, 0.6916884573920534)

    def test_random2(self):
        dm = np.array([
            [ (0.333204+0j),(-0.1112-0.113795j),(-0.099827+0.069346j),(-0.002388-0.022364j) ],
            [ (-0.1112+0.113795j),(0.196206+0j),(0.001052-0.11965j),(0.111748-0.009399j) ],
            [ (-0.099827-0.069346j),(0.001052+0.11965j),(0.180806+0j),(0.088287+0.120957j) ],
            [ (-0.002388+0.022364j),(0.111748+0.009399j),(0.088287-0.120957j),(0.289469+0j) ],
        ], dtype=np.complex128)
        self.check_entropy(dm, 0.9691946314869655)

class Renyi(ut.TestCase):

    def check_entropy(self, dm, correct):
        '''
        Check correctness.

        dm : density matrix
        correct : list of tuples (n, value), where n is the exponent for the Renyi
            entropy, and value is the correct entropy for that exponent
        '''
        eps = 1E-14
        for n, val in correct:
            with self.subTest(n=n):

                with self.subTest(method='eigsolve'):
                    check = dm_renyi_entropy(dm, n, 'eigsolve')
                    self.assertTrue(np.isclose(check, val, rtol=0, atol=eps),
                                    msg = '\ncheck: %s\ncorrect: %s' % (str(check), str(val)))

                with self.subTest(method='matrix_power'):
                    if n == 'inf' or int(n) == n:
                        check = dm_renyi_entropy(dm, n, 'matrix_power')
                        self.assertTrue(np.isclose(check, val, rtol=0, atol=eps),
                                        msg = '\ncheck: %s\ncorrect: %s' % (str(check), str(val)))
                    else:
                        with self.assertRaises(TypeError):
                            check = dm_renyi_entropy(dm, n, 'matrix_power')

    def test_pure_state(self):
        dm = np.array([[1, 0], [0, 0]], dtype=float)

        tests = [
            (0,     0.0),
            (1,     0.0),
            (2,     0.0),
            (2.0,   0.0),
            (2.5,   0.0),
            ('inf', 0.0),
        ]

        self.check_entropy(dm, tests)

    def test_pure_X_state(self):
        dm = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        tests = [
            (0,     0.0),
            (1,     0.0),
            (2,     0.0),
            (2.0,   0.0),
            (2.5,   0.0),
            ('inf', 0.0),
        ]
        self.check_entropy(dm, tests)

    def test_pure_Y_state(self):
        dm = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
        tests = [
            (0,     0.0),
            (1,     0.0),
            (2,     0.0),
            (2.0,   0.0),
            (2.5,   0.0),
            ('inf', 0.0),
        ]
        self.check_entropy(dm, tests)

    def test_diag_mixed(self):
        dm = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
        tests = [
            (0,     np.log(2)),
            (1,     np.log(2)),
            (2,     np.log(2)),
            (2.0,   np.log(2)),
            (2.5,   np.log(2)),
            ('inf', np.log(2)),
        ]
        self.check_entropy(dm, tests)

    def test_diag_mixed2(self):
        dm = np.array([[3/4, 0], [0, 1/4]], dtype=np.complex128)
        tests = [
            (0,     np.log(2)),
            (1,     -(3/4)*np.log(3)+np.log(4)),
            (2,     np.log(8/5)),
            (2.0,   np.log(8/5)),
            (2.5,   0.43801919639725084),
            ('inf', np.log(4/3)),
        ]
        self.check_entropy(dm, tests)

    def test_diag_mixed3(self):
        dm =  (3/4) * np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
        dm += (1/4) * np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
        tests = [
            (0,     np.log(2)),
            (1,     -(3/4)*np.log(3)+np.log(4)),
            (2,     np.log(8/5)),
            (2.0,   np.log(8/5)),
            (2.5,   0.43801919639725084),
            ('inf', np.log(4/3)),
        ]
        self.check_entropy(dm, tests)

if __name__ == '__main__':
    ut.main()
