'''
This file generates test cases for the density matrix and entropy computations in dynamite.
'''

import numpy as np
import qutip as qtp

STATE_ROUND = 3
MAX_WIDTH = 70

def generate_qtp_state(L):
    '''
    Generate a random state in QuTiP with the right tensor product structure.
    '''
    rand_state = np.random.uniform(-1, 1, 2**L) + 1j*np.random.uniform(-1, 1, 2**L)
    # round to make printing easier
    rand_state /= np.linalg.norm(rand_state)
    rand_state = np.around(rand_state, decimals=STATE_ROUND)
    rtn = qtp.Qobj(rand_state, dims=[[2]*L, [1]*L])
    return rtn

def get_density_matrix(state, keep):
    '''
    Compute the reduced density matrix of a QuTiP state, keeping the spins in 'keep'.
    '''
    full_dm = state * state.dag()
    # the indexing in qutip is the opposite of in dynamite
    # so we have to change the indices
    L = len(state.dims[0])
    keep = [L-x-1 for x in keep]

    red_dm = full_dm.ptrace(keep)
    return red_dm

def get_ent_entropy(dm):
    '''
    Compute the Von Neumann entropy of the mixed state dm.
    '''
    return qtp.entropy_vn(dm)


if __name__ == '__main__':

    test_cases = [
        {
            'L' : 4,
            'keep' : [
                [0],
                [0, 2],
                [1, 3],
                [2]
            ]
        },
        {
            'L' : 5,
            'keep' : [
                [0, 2, 4],
                [1, 3],
            ]
        }
    ]

    np.random.seed(0)

    for d in test_cases:
        state = generate_qtp_state(d['L'])
        print('======')
        print('L,%d' % d['L'])

        state_str = ''
        line_len = 0
        for x in state.full().flatten():
            x_str = str(x)+','
            state_str += x_str
            line_len += len(x_str)
            if line_len > MAX_WIDTH:
                state_str += '\n'
                line_len = 0
        print(state_str)

        for keep in d['keep']:
            dm = get_density_matrix(state, keep)
            ee = get_ent_entropy(dm)
            print()
            print('keep,%s' % str(keep))

            full_dm = np.around(dm.full(), decimals=2*STATE_ROUND)
            print('dm:')
            for row in full_dm:
                print('[',','.join(str(x) for x in row),'],')

            print('EE,%s' % str(ee))
        print()
