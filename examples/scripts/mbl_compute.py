'''
Run the compute stage of the MBL computation from mbl.py, outputting the
results to stdout. The results can be read and plotted using
mbl_plot.py.
'''

# we are going to write all our other output to stderr to not pollute stdout
from sys import stderr
import numpy as np

from mbl import parse_command_line, compute_subspace, build_hamiltonian
from mbl import compute_level_stats

from dynamite import config
from dynamite.tools import mpi_print

def print_results(results, disorder_strengths):
    mpi_print(*disorder_strengths, sep=',')
    mpi_print()

    for row in results:
        mpi_print(*row, sep=',')

def main():
    params = parse_command_line()

    # globally set the system size
    config.L = params.L

    disorder_strengths = np.linspace(params.min_w, params.max_w, params.npoints)
    subspace = compute_subspace()

    results = np.zeros((params.npoints, params.npoints), dtype=float)

    for n,w in enumerate(disorder_strengths):
        msg = 'Beginning disorder strength {w} ({n}/{npoints})'
        mpi_print(msg.format(w=w, n=n+1, npoints=params.npoints), file=stderr)
        for i in range(params.realizations):
            mpi_print('Computing disorder realization', i, file=stderr)
            H = build_hamiltonian(w, n*params.npoints + i)
            H.subspace = subspace
            results[n] += compute_level_stats(H, params.npoints, params.nev)
        mpi_print(file=stderr)

    results /= params.realizations

    print_results(results, disorder_strengths)

if __name__ == "__main__":
    main()
