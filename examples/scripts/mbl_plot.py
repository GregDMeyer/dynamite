'''
Plot data from a file containing the output of mbl_compute.py (or stdin)
'''

import fileinput
import numpy as np
from mbl import plot_results

def read_data():
    data = fileinput.input()
    first_line = next(data)
    disorder_strengths = np.fromstring(first_line, sep=',')
    next(data) # next line is blank

    npoints = len(disorder_strengths)

    results = np.ndarray((npoints, npoints))
    for n, line in enumerate(data):
        results[n] = np.fromstring(line, sep=',')

    return results, disorder_strengths

def main():
    results, disorder_strengths = read_data()
    plot_results(results, disorder_strengths)

if __name__ == '__main__':
    main()
