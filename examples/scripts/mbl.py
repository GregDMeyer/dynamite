'''
This example performs an analysis of the level statistics of many-body localization.
'''

import numpy as np
import argparse
from matplotlib import pyplot as plt

from dynamite import config
from dynamite.operators import sigma_plus, sigma_minus, sigmaz, index_sum, op_sum
from dynamite.subspaces import Auto
from dynamite.tools import mpi_print

def build_hamiltonian(w, seed):
    '''
    Build the Heisenberg + random field Hamiltonian.

    Parameters
    ----------
    w : float
        The magnitude of the disorder

    seed : int
        The seed for the RNG for the disorder
    '''

    # this normalization is because these are sigma matrices, not spin matrices
    hopping = 1/8 * (sigma_plus(0)*sigma_minus(1) + sigma_minus(0)*sigma_plus(1))

    # add ZZ interaction
    ZZ = 1/2 * sigmaz(0)*sigmaz(1)

    # translate the hopping and ZZ terms along the whole spin chain
    # we don't need to specify spin chain length because we set it with config.L
    H = index_sum(hopping + ZZ)

    # we have to be careful that the same values are generated on each MPI process!
    # that's why we explicitly seed the random number generator
    np.random.seed(seed)
    disorder = np.random.uniform(-w, w, H.get_length())

    H += op_sum((1/2)*h_i*sigmaz(i) for i,h_i in enumerate(disorder))

    return H

def compute_subspace():
    '''
    Compute the subspace corresponding to the total magnetization zero sector.
    '''
    H = build_hamiltonian(0, 0)
    half_chain = H.get_length() // 2

    # our state is half up, half down (or close, if we have odd # spins)
    state = 'U'*half_chain + 'D'*(H.get_length() - half_chain)

    # compute the subspace of the Hamiltonian that contains this state
    # for this Hamiltonian, it's total magnetization conservation
    subspace = Auto(H, state)

    return subspace

def compute_level_stats(H, npoints, nev):
    '''
    Compute level statistics at various points in the spectrum.

    Parameters
    ----------
    H : dynamite.operators.Operator
        The Hamiltonian

    npoints : int
        The number of points in the spectrum at which to sample

    nev : int
        The number of eigenvalues to compute at each point in the spectrum
    '''

    results = np.zeros(npoints)

    # first compute the ends of the spectrum
    low_evals = H.eigsolve(which='smallest', nev=nev)
    high_evals = H.eigsolve(which='largest', nev=nev)

    results[0] = compute_avg_r_ratio(low_evals)
    results[-1] = compute_avg_r_ratio(high_evals)

    interior_points = np.linspace(
        np.mean(low_evals),
        np.mean(high_evals),
        npoints
    )[1:-1]

    for n,t in enumerate(interior_points):
        evals = H.eigsolve(target=t, nev=nev)
        results[n+1] = compute_avg_r_ratio(evals)

    return results

def compute_avg_r_ratio(evals):
    '''
    Compute the average ratio of adjacent gaps in eigenvalues.
    '''

    r = 0
    diffs = np.abs(np.diff(evals))

    for i in range(len(diffs)-1):
        r += min(diffs[i], diffs[i+1]) / max(diffs[i], diffs[i+1])

    return r / (len(diffs)-1)

def gen_plot_bounds(ary):
    '''
    Generate intervals centered on the points in ary.
    (This function is just a helper for plotting the heat map).
    '''
    delta = 0.5*(ary[1] - ary[0])
    return np.hstack([ary-delta, [ary[-1]+delta]])

def plot_results(results, disorder_strengths):
    '''
    Generate a line plot and heat-map plot of the results.
    '''
    f, (lineax, heatax) = plt.subplots(2,1)

    # plot our line of average ratio vs disorder
    lineax.plot(disorder_strengths, np.mean(results, axis=1))

    # make a heat map plot of the data to see dependence on energy
    energy_points = np.linspace(0, 1, len(disorder_strengths))

    y, x = np.meshgrid(
        gen_plot_bounds(energy_points),
        gen_plot_bounds(disorder_strengths)
    )
    c = heatax.pcolormesh(x, y, results)
    plt.xlabel('disorder')
    plt.ylabel('epsilon')
    f.colorbar(c, ax=heatax)
    plt.show()

def parse_command_line(argv=None):
    '''
    Parse relevant options from the command line.
    '''

    description = 'Compute level statistics of a localizing Hamiltonian ' +\
                  'for various disorder strengths.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--min_w', type=float, default=0.5,
                        help='The lower bound on disorder strength')

    parser.add_argument('--max_w', type=float, default=5.0,
                        help='The upper bound on disorder strength')

    parser.add_argument('--npoints','-n', type=int, default=10,
                        help='The number of points to sample in disorder and energy space.')

    parser.add_argument('--realizations', '-r', type=int, default=3,
                        help='The number of times to disorder average')

    parser.add_argument('-L', type=int, default=10,
                        help='The system size.')

    parser.add_argument('--nev', type=int, default=64,
                        help='The number of eigenvalues to compute at each point in the spectrum.')

    args = parser.parse_args(argv)

    return args

def main():

    params = parse_command_line()

    # globally set the system size
    config.L = params.L

    disorder_strengths = np.linspace(params.min_w, params.max_w, params.npoints)
    subspace = compute_subspace()

    results = np.zeros((params.npoints, params.npoints), dtype=float)

    for n,w in enumerate(disorder_strengths):
        msg = 'Beginning disorder strength {w} ({n}/{npoints})'
        mpi_print(msg.format(w=w, n=n+1, npoints=params.npoints))
        for i in range(params.realizations):
            mpi_print('Computing disorder realization', i)
            H = build_hamiltonian(w, n*params.npoints + i)
            H.subspace = subspace
            results[n] += compute_level_stats(H, params.npoints, params.nev)
        mpi_print()

    results /= params.realizations

    plot_results(results, disorder_strengths)

if __name__ == '__main__':
    main()
