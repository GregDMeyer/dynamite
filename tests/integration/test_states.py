'''
Integration tests for states.
'''

import math
import numpy as np

import dynamite_test_runner as dtr

from dynamite import config
from dynamite.states import State, UninitializedError
from dynamite.subspaces import Parity, SpinConserve, Auto
from dynamite.operators import sigmaz, sigmax, sigmay, index_sum
from dynamite.computations import reduced_density_matrix

class RandomSeed(dtr.DynamiteTestCase):

    def test_generation(self):
        '''
        Make sure that different processors get the same random seed.
        '''

        from dynamite import config
        config._initialize()
        from petsc4py import PETSc

        seed = State.generate_time_seed()

        if PETSc.COMM_WORLD.size > 1:
            comm = PETSc.COMM_WORLD.tompi4py()
            all_seeds = comm.gather(seed, root=0)
        else:
            all_seeds = [seed]

        if PETSc.COMM_WORLD.rank == 0:
            self.assertTrue(all(s == seed for s in all_seeds))

class ToNumpy(dtr.DynamiteTestCase):

    def setUp(self):
        from petsc4py import PETSc

        self.v = PETSc.Vec().create()
        self.v.setSizes(PETSc.COMM_WORLD.size)
        self.v.setFromOptions()
        self.v.set(-1)

        self.v[PETSc.COMM_WORLD.rank] = PETSc.COMM_WORLD.rank
        self.v.assemblyBegin()
        self.v.assemblyEnd()

    def test_to_zero(self):
        from petsc4py import PETSc
        npvec = State._to_numpy(self.v)
        if PETSc.COMM_WORLD.rank == 0:
            for i in range(PETSc.COMM_WORLD.rank):
                self.assertTrue(npvec[i] == i)
        else:
            self.assertIs(npvec, None)

    def test_to_all(self):
        from petsc4py import PETSc
        npvec = State._to_numpy(self.v, to_all = True)
        for i in range(PETSc.COMM_WORLD.rank):
            self.assertTrue(npvec[i] == i)

class PetscMethods(dtr.DynamiteTestCase):
    '''
    Tests that the methods directly included from PETSc function as intended.
    '''
    def test_norm(self):
        state = State()
        start, end = state.vec.getOwnershipRange()
        state.vec[start:end] = np.array([1]*(end-start))
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        state.set_initialized()
        self.assertAlmostEqual(state.norm()**2, state.subspace.get_dimension())

    def test_normalize(self):
        state = State()
        start, end = state.vec.getOwnershipRange()
        state.vec[start:end] = np.array([1]*(end-start))
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        state.set_initialized()

        state.normalize()
        self.assertTrue(state.norm() == 1)

    def test_copy_preallocate(self):
        state1 = State()
        state2 = State()
        start, end = state1.vec.getOwnershipRange()
        state1.vec[start:end] = np.arange(start, end)
        state1.vec.assemblyBegin()
        state1.vec.assemblyEnd()
        state1.set_initialized()

        result = np.ndarray((end-start,), dtype=np.complex128)
        state1.copy(state2)
        result[:] = state2.vec[start:end]

        self.assertTrue(np.array_equal(result, np.arange(start, end)))

    def test_copy_exception_L(self):
        state1 = State(subspace=Parity('even'))
        state2 = State(subspace=Parity('odd'))

        with self.assertRaises(ValueError):
            state1.copy(state2)

    def test_copy_nopreallocate(self):
        state1 = State()
        start, end = state1.vec.getOwnershipRange()
        state1.vec[start:end] = np.arange(start, end)
        state1.vec.assemblyBegin()
        state1.vec.assemblyEnd()
        state1.set_initialized()

        result = np.ndarray((end-start,), dtype=np.complex128)
        state2 = state1.copy()
        result[:] = state2.vec[start:end]

        self.assertTrue(np.array_equal(result, np.arange(start, end)))

    def check_local_vector(self, result_vec, fn, *arg_vecs):
        start, end = result_vec.getOwnershipRange()
        local_arg_vecs = []
        for vec in arg_vecs:
            local_arg_vecs.append(
                np.array(vec[start:end])
            )

        # apply the function (in a vectorized manner)
        # to the local portion of each arg_vec
        correct_local_vec = fn(*local_arg_vecs)

        # machine epsilon, but with a bunch of operations on top of it
        tol_scale = 5E-12

        tol = tol_scale*np.abs(correct_local_vec)
        diffs = np.abs(result_vec[start:end]-correct_local_vec)

        max_diff_idx = np.argmax(diffs/tol)
        msg = f'failed at index {max_diff_idx}: '
        msg += f'{diffs[max_diff_idx]} > {tol[max_diff_idx]} '
        msg += f'(abs({tol_scale}*{correct_local_vec[max_diff_idx]}))'
        self.assertTrue(np.all(diffs < tol), msg=msg)

    def test_imul(self):
        vals = [2, 3.14]
        for val in vals:
            with self.subTest(val=val):
                state = State(state='random')
                orig = state.copy()

                state *= val

                self.check_local_vector(
                    state.vec,
                    lambda x: val*x,
                    orig.vec
                )

    def test_idiv(self):
        val = 3.14
        state = State(state='random')
        orig = state.copy()

        state /= val

        inv_val = 1/val
        self.check_local_vector(
            state.vec,
            lambda x: inv_val*x,
            orig.vec
        )

    def test_imul_exception_ary(self):
        val = np.array([3.1, 4])
        state = State(state='U'*config.L)
        with self.assertRaises(TypeError):
            state *= val

    def test_imul_exception_vec(self):
        state1 = State(state='U'*config.L)
        state2 = State(state='U'*config.L)
        with self.assertRaises(TypeError):
            state1 *= state2

    def test_scale(self):
        vals = [2, 3.14, 0.5j]
        for val in vals:
            with self.subTest(val=val):
                state = State(state='random')
                orig = state.copy()
                state.scale(val)
                self.check_local_vector(
                    state.vec,
                    lambda x: val*x,
                    orig.vec
                )

    def test_axpy(self):
        alphas = [1.0, 3.14, 0.5j]
        for alpha in alphas:
            with self.subTest(alpha=alpha):
                x = State(state='random')
                y = State(state='random')
                orig_y = y.copy()

                y.axpy(alpha, x)

                self.check_local_vector(
                    y.vec,
                    lambda x, y: y + alpha*x,
                    x.vec,
                    orig_y.vec
                )

    def test_scale_and_sum(self):
        alphas = [1.0, 3.14, 0.5j]
        for alpha in alphas:
            for beta in alphas:
                with self.subTest(alpha=alpha, beta=beta):
                    x = State(state='random')
                    y = State(state='random')
                    orig_y = y.copy()

                    y.scale_and_sum(alpha, beta, x)

                    self.check_local_vector(
                        y.vec,
                        lambda x, y: alpha*x + beta*y,
                        x.vec,
                        orig_y.vec
                    )

    def test_scale_and_sum_same(self):
        x = State(state='random')
        with self.assertRaises(ValueError):
            x.scale_and_sum(1, 1, x)

    def test_iadd_vector(self):
        x = State(state='random')
        y = State(state='random')
        orig_y = y.copy()

        y += x

        self.check_local_vector(
            y.vec,
            lambda x, y: x + y,
            x.vec,
            orig_y.vec
        )

    def test_iadd_value(self):
        vals = [3.14, 0.5j]
        for val in vals:
            with self.subTest(val=val):
                state = State(state='random')
                orig = state.copy()
                state += val
                self.check_local_vector(
                    state.vec,
                    lambda x: val+x,
                    orig.vec
                )

    def test_add_vector(self):
        x = State(state='random')
        y = State(state='random')

        z = x+y

        self.check_local_vector(
            z.vec,
            lambda x, y: x + y,
            x.vec,
            y.vec
        )

    def test_add_value(self):
        val = 3.14
        state = State(state='random')
        result = state + val
        self.check_local_vector(
            result.vec,
            lambda x: val+x,
            state.vec
        )

    def test_radd_value(self):
        val = 3.14
        state = State(state='random')
        result = val + state
        self.check_local_vector(
            result.vec,
            lambda x: val+x,
            state.vec
        )

    def test_add_subspace_fail(self):
        x = State(state='random', subspace=Parity('odd'))
        y = State(state='random', subspace=Parity('even'))
        with self.assertRaises(ValueError):
            y += x

    def test_isub_vector(self):
        x = State(state='random')
        y = State(state='random')
        orig_y = y.copy()

        y -= x

        self.check_local_vector(
            y.vec,
            lambda x, y: y-x,
            x.vec,
            orig_y.vec
        )

    def test_isub_value(self):
        vals = [3.14, 0.5j]
        for val in vals:
            with self.subTest(val=val):
                state = State(state='random')
                orig = state.copy()
                state -= val
                self.check_local_vector(
                    state.vec,
                    lambda x: x-val,
                    orig.vec
                )

    def test_sub_vector(self):
        x = State(state='random')
        y = State(state='random')

        z = x-y

        self.check_local_vector(
            z.vec,
            lambda x, y: x - y,
            x.vec,
            y.vec
        )

    def test_sub_value(self):
        val = 3.14
        state = State(state='random')
        result = state - val
        self.check_local_vector(
            result.vec,
            lambda x: x-val,
            state.vec
        )

    def test_rsub_value(self):
        val = 3.14
        state = State(state='random')
        result = val - state
        self.check_local_vector(
            result.vec,
            lambda x: val-x,
            state.vec
        )

    def test_mul_vector(self):
        x = State(state='random')
        y = State(state='random')

        with self.assertRaises(TypeError):
            x*y

    def test_mul_value(self):
        val = 3.14
        state = State(state='random')
        result = state * val
        self.check_local_vector(
            result.vec,
            lambda x: val*x,
            state.vec
        )

    def test_rmul_value(self):
        val = 3.14
        state = State(state='random')
        result = val * state
        self.check_local_vector(
            result.vec,
            lambda x: val*x,
            state.vec
        )


class Projection(dtr.DynamiteTestCase):

    def check_projection(self, state, idx):
        for val in (0, 1):
            with self.subTest(value=val):
                proj_state = state.copy()

                proj_state.project(idx, val)

                sz = sigmaz(idx)
                sz.add_subspace(state.subspace)

                self.assertAlmostEqual(
                    proj_state.dot(sz*proj_state),
                    1 if val == 0 else -1
                )

    def test_full(self):
        state = State(state='random')

        for idx in [0, config.L//2, config.L-1]:
            with self.subTest(idx=idx):
                self.check_projection(state, idx)

    def test_parity(self):
        for parity in ['odd', 'even']:
            with self.subTest(parity=parity):

                state = State(subspace=Parity(parity),
                              state='random')
                for idx in [0, config.L//2, config.L-1]:
                    with self.subTest(idx=idx):
                        self.check_projection(state, idx)

    def test_index_exceptions(self):
        state = State(state='random')
        for idx in [-1, config.L, config.L+1]:
            with self.subTest(idx=idx):
                with self.assertRaises(ValueError):
                    state.project(idx, 0)

    def test_value_exception(self):
        state = State(state='random')
        with self.assertRaises(ValueError):
            state.project(0, -1)


class Saving(dtr.DynamiteTestCase):

    fname = '/tmp/dnm_test_save'

    def check_states_equal(self, a, b):
        self.assertTrue(a.subspace.identical(b.subspace))
        self.assertTrue(a.vec.equal(b.vec))

    def test_save_simple(self):
        state = State(state='random')
        state.save(self.fname)
        loaded = State.from_file(self.fname)
        self.check_states_equal(state, loaded)

    def test_save_parity(self):
        subspace = Parity('even')
        state = State(state='random', subspace=subspace)
        state.save(self.fname)
        loaded = State.from_file(self.fname)
        self.check_states_equal(state, loaded)

    def test_save_spinconserve(self):
        subspace = SpinConserve(config.L, config.L//2)
        state = State(state='random', subspace=subspace)
        state.save(self.fname)
        loaded = State.from_file(self.fname)
        self.check_states_equal(state, loaded)

    def test_save_spinconserve_spinflip(self):
        subspace = SpinConserve(config.L, config.L//2, spinflip=True)
        state = State(state='random', subspace=subspace)
        state.save(self.fname)
        loaded = State.from_file(self.fname)
        self.check_states_equal(state, loaded)

    def test_save_auto(self):
        H = index_sum(sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1))

        half_L = config.L//2
        subspace = Auto(H, 'U'*half_L + 'D'*(config.L - half_L))
        state = State(state='random', subspace=subspace)
        state.save(self.fname)
        loaded = State.from_file(self.fname)
        self.check_states_equal(state, loaded)


class Uninitialized(dtr.DynamiteTestCase):

    def test_copy_neither(self):
        s0 = State()
        s1 = s0.copy()
        self.assertFalse(s1.initialized)

    def test_copy_from_uninit(self):
        s0 = State()
        s1 = State(state='U'*config.L)
        with self.assertRaises(UninitializedError):
            s0.copy(s1)

    def test_copy_to_uninit(self):
        s0 = State(state='U'*config.L)
        s1 = State()
        s0.copy(s1)
        self.assertTrue(s1.initialized)

    def test_project(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            s0.project(0, 0)

    def test_to_numpy(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            s0.to_numpy()

    def test_dot_from(self):
        s0 = State()
        s1 = State(state='U'*config.L)
        with self.assertRaises(UninitializedError):
            s0.dot(s1)

    def test_dot_to(self):
        s0 = State(state='U'*config.L)
        s1 = State()
        with self.assertRaises(UninitializedError):
            s0.dot(s1)

    def test_norm(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            s0.norm()

    def test_normalize(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            s0.normalize()

    def test_imul(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            s0 *= 2

    def test_idiv(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            s0 /= 2

    def test_op_dot(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            sigmax().dot(s0)

    def test_op_evolve(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            sigmax().evolve(s0, t=1.0)

    def test_rdm(self):
        s0 = State()
        with self.assertRaises(UninitializedError):
            reduced_density_matrix(s0, [])


class StringMPI(dtr.DynamiteTestCase):

    def test_get_nonzero_elements(self):
        s = State()
        s.vec.set(0.1)
        s.set_initialized()

        check = s._get_nonzero_elements()

        correct = [
            (0, 0.1),
            (1, 0.1),
            (2, 0.1),
            (0, 0),  # marks omitted values
            (len(s)-1, 0.1),
        ]

        self.assertTrue(
            all(
                idx == correct_idx and val == correct_val
                for ((idx, val), (correct_idx, correct_val)) in zip(check, correct)
            ),
            msg=f'\ncheck:\n{check}\n\ncorrect:\n{correct}'
        )

    def test_get_nonzero_elements_middle(self):
        s = State(state='0'*(config.L-1)+'1')  # middle element
        result = s._get_nonzero_elements()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], len(s)//2)
        self.assertEqual(result[0][1], 1)

    def test_get_nonzero_elements_middle_many(self):
        s = State()
        start, end = s.vec.getOwnershipRange()
        dim = s.subspace.get_dimension()

        fill_start = max(start, dim//4)
        fill_end = min(end, 3*dim//4)

        if fill_start < fill_end:
            s.vec[fill_start:fill_end] = 0.001*np.arange(fill_start, fill_end)

        s.vec.assemblyBegin()
        s.vec.assemblyEnd()
        s.set_initialized()

        check = s._get_nonzero_elements()

        nonzero_start = dim//4
        nonzero_end = 3*dim//4
        correct = [
            (nonzero_start, 0.001*nonzero_start),
            (nonzero_start+1, 0.001*(nonzero_start+1)),
            (nonzero_start+2, 0.001*(nonzero_start+2)),
            (0, 0),  # marks omitted values
            (nonzero_end-1, 0.001*(nonzero_end-1)),
        ]

        self.assertTrue(
            all(
                idx == correct_idx and val == correct_val
                for ((idx, val), (correct_idx, correct_val)) in zip(check, correct)
            ),
            msg=f'\ncheck:\n{check}\n\ncorrect:\n{correct}'
        )


class FullStateSetting(dtr.DynamiteTestCase):

    def test_constant(self):
        s = State()

        def val_fn(idx):
            return np.full(idx.shape, 0.1)

        start, end = s.vec.getOwnershipRange()

        for vectorize in (True, False):
            with self.subTest(vectorize=vectorize):
                s.set_all_by_function(val_fn, vectorize=vectorize)

                self.assertTrue(all(v == 0.1 for v in s.vec[start:end]))

    def test_val(self):
        s = State()

        def val_fn(idx):
            return 0.001*idx

        start, end = s.vec.getOwnershipRange()

        for vectorize in (True, False):
            with self.subTest(vectorize=vectorize):
                s.set_all_by_function(val_fn, vectorize=vectorize)

                self.assertTrue(all(
                    v == (start+i)*0.001
                    for i, v in enumerate(s.vec[start:end])
                ))

    def test_phase(self):
        s = State()

        def val_fn(idx):
            return 0.01*np.exp(1j*idx)

        start, end = s.vec.getOwnershipRange()

        for vectorize in (True, False):
            with self.subTest(vectorize=vectorize):
                s.set_all_by_function(val_fn, vectorize=vectorize)

                self.assertTrue(all(
                    v == val_fn(start+i)
                    for i, v in enumerate(s.vec[start:end])
                ))

    def test_subspace(self):
        s = State(subspace=SpinConserve(config.L, config.L//2))

        def val_fn(state):
            # 1 if last spin is down, 0 if it's up
            return (state >> (config.L-1)) & 1

        one_start = math.comb(config.L-1, config.L//2)

        start, end = s.vec.getOwnershipRange()

        for vectorize in (True, False):
            with self.subTest(vectorize=vectorize):
                s.set_all_by_function(val_fn, vectorize=vectorize)

                for i, v in enumerate(s.vec[start:end]):
                    if (i+start) >= one_start:
                        if v != 1:
                            msg = f'value {v} at index {i+start}'
                            break
                    elif v != 0:
                        msg = f'value {v} at index {i+start}'
                        break
                else:
                    msg = ''

                self.assertTrue(
                    not msg,
                    msg=msg,
                )


if __name__ == '__main__':
    dtr.main()
