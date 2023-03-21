# Many-body localization

## In this example

 - Eigensolving for ground states
 - Eigensolving for states in the middle of the spectrum
 - The `SpinConserve` subspace
 - Computing entanglement entropy
 - Coordinating randomness across MPI ranks

## Overview

This project uses dynamite to explore many-body localization (MBL), a surprising phenomenon in which certain many-body systems with sufficiently strong disorder fail to thermalize. In particular we will study the *MBL transition*, in which the system moves from thermalizing to localized as the disorder strength is increased. Characterization of this transition has remained elusive: finite size effects seem hard to avoid, and tensor network methods break down due to extensive entanglement in states near the transition. Thus, iterative methods like the Krylov subspace methods used by dynamite have proved to be one of the best tools for its study.[<sup>1</sup>](#ref1) We refer readers interested in learning more about the physics of MBL to one of the excellent review papers on the subject.[<sup>2</sup>](#ref2)

In this project we explore a model of nearest-neighbor Heisenberg interactions on a 1D chain, with disorder implemented as random Z fields on each site:

$$H = \sum_{\left<i,j\right>} \vec{S}_i \cdot \vec{S}_j + \sum_i h_i S^z_i$$

where $\vec{S} = (S^x, S^y, S^z)$, the subscripts indicate the index of the spin in the chain, and the angle brackets indicate that the indices run over nearest neighbors. The values of $h_i$ are drawn from a uniform distribution $\left[-h, h\right]$ where $h$ is a parameter that controls the strength of the disorder.

In the script `run_mbl.py` this is implemented by the `build_hamiltonian` function:


```python
from dynamite import config
config.L = 10

from run_mbl import build_hamiltonian
build_hamiltonian(h=2)
```




$\sum_{i=0}^{8}0.25\left(\sigma^x_{i}\sigma^x_{i+1} + \sigma^y_{i}\sigma^y_{i+1} + \sigma^z_{i}\sigma^z_{i+1}\right) + -0.11\sigma^z_{0} + 0.699\sigma^z_{1} + 0.428\sigma^z_{2} + -0.081\sigma^z_{3} + 0.413\sigma^z_{4} + 0.454\sigma^z_{5} + -0.026\sigma^z_{6} + 0.661\sigma^z_{7} + -0.883\sigma^z_{8} + -0.957\sigma^z_{9}$



## Goals

In this project we plot two quantities that help us identify the MBL transition: the half-chain entanglement entropy of eigenstates, and an eigenvalue statistic called the *adjacent gap ratio*. The half chain entanglement entropy $S_{L/2}$ is simply the bipartite von Neumann entropy when half the spins are traced out. The MBL transition should correspond to a transition from volume law to area law entanglement. The adjacent gap ratio computes a measure of how likely eigenvalues are to be near each other. Let $\Delta_i$ be the gap between the $i^\mathrm{th}$ eigenvalue and the following one; that is, $\Delta_i = \lambda_{i+1} - \lambda_i$. Then, the adjacent gap ratio is defined as $r_i = \frac{\min \left( \Delta_i, \Delta_{i+1} \right)}{\max \left( \Delta_i, \Delta_{i+1} \right)}$. Random matrix theory tells us that in the thermal phase, the expectation value of $r$ should be $\left< r \right> \approx 0.53$, while in the localized phase $\left< r \right> \approx 0.39$.

The key feature that makes MBL so interesting is that the transition from volume to area law entanglement does not only occur in the ground state, but in excited states as well. This presents a great use case for dynamite's `target` eigenvalue solver, which finds the $k$ eigenvalues (and eigenvectors) closest to a target energy, where $k$ is user configurable. So, the plan is as follows: we will choose a few energies at various points in the spectrum, solve for some reasonable number (say, 32) eigenpairs near each of those points, and then compute the entanglement entropy and adjacent gap ratio for all of those eigenpairs.

## Remark: details of the solver for excited states

The iterative solver used by dynamite to solve for eigenpairs is very good at finding *extremal* eigenvalues---those with the largest absolute value. To apply this solver to finding sets of interior eigenvalues, dynamite uses what's called the *shift-invert* transformation. Instead of applying the solver to the Hamiltonian $H$ itself, it is applied to the transformed Hamiltonian $(H-E_\mathrm{targ})^{-1}$, where $E_\mathrm{targ}$ is the target energy near which eigenpairs are desired. With this transformation, the eigenvalues closest to $E_\mathrm{targ}$ become extremal.

To apply the iterative solver, we need to be able to efficiently compute matrix vector products of the form $(H-E_\mathrm{targ})^{-1} \vec{x}$. To implement this, PETSc (the linear algebra library that underlies dynamite) computes the LU factorization of $H$---a pair of matrices $L$ and $U$ where $L$ is lower triangular and $U$ is upper triangular, and $H = LU$. This factorization makes it very easy to perform the linear solves needed to apply the matrix inverse, but it comes with multiple costs. For one, the $LU$ factorization generally requires significantly more memory to store than $H$ itself. Furthermore, computing the $LU$ factorization can be computationally expensive.

Thus, you will find solving for interior eigenvalues to be much more computationally intensive than solving for ground states.

## Remark: disorder realizations and parallelism

(also discussed in the SYK example)

The MBL Hamiltonian is a case in which getting good data requires disorder averaging---that is, running the computation many times with fresh randomness. Given $N$ CPU cores there are two broad ways one can parallelize that process: (1) running $N$ disorder realizations independently at the same time, each using one core, and (2) using MPI to parallelize one computation across all $N$ cores and then doing each disorder realization in sequence. In this case, (1) will almost always be faster and should be prioritized---while the MPI parallelism in dynamite is highly optimized, there will always be some cost to the communication between cores.

However, there are situations in which using MPI may be preferable, for example if running $N$ independent disorder realizations uses too much memory. Ultimately, the user should experiment with different configurations to determine what gives the best performance. Ideally, in practice one would simply make use of a large cluster of GPUs, running independent disorder realizations on each one.

## Remark: using randomness in dynamite

(also discussed in SYK example)

One needs to take extra care when using randomness in code that will be run under MPI with multiple ranks. Each rank is its own Python process, and by default will have a different random seed---so if you are not careful, each of your ranks may end up trying to build a different Hamiltonian! (dynamite does check that the Hamiltonian is the same on all ranks before building the underlying matrix, so you will get an error if this happens).

There are two ways to handle this: one is to have rank 0 pick a random seed and use MPI to communicate it to all the other ranks, and the other is to simply pass a seed on the command line. Both are implemented in this example for demonstration purposes: if no seed is passed on the command line, then one is generated and communicated to all MPI ranks. If you pass a random seed on the command line, make sure to change it each time you run the code if you want new disorder realizations!

**Note 1:** when setting a random state via `State(state='random')`, dynamite is already careful about coordinating randomness between MPI ranks, so the user does not need to worry about it in that case.

**Note 2:** If you will never run your code on multiple MPI ranks, you don't need to worry about this at all. In particular, running on a GPU with 1 CPU process will not encounter this issue.

## Usage

The computation is implemented in `run_mbl.py`. The script will output, in CSV format, statistics of eigenpairs clustered around equally-spaced points throughout the spectrum. 

This example also includes a script `plot_result.py` which can be used to plot the results using matplotlib. Data can either be piped directly to it:
```bash
python run_mbl.py -L 14 | python plot_result.py
```
or one can save the data to a file and pass the filename on the command line:
```bash
python run_mbl.py -L 14 > output.csv
python plot_result.py output.csv
```

We also provide an example output file `example_output.csv` which was run with... # TODO: update details when job finishes

Here are the command line options for `run_mbl.py`:


```python
! python run_mbl.py -h
```

    usage: run_mbl.py [-h] -L L [--seed SEED] [--iters ITERS]
                      [--energy-points ENERGY_POINTS] [--h-points H_POINTS]
                      [--h-min H_MIN] [--h-max H_MAX] [--nev NEV]
    
    options:
      -h, --help            show this help message and exit
      -L L                  spin chain length
      --seed SEED           seed for random number generator. if omitted, a random
                            seed is chosen by querying system hardware randomness
      --iters ITERS         number of disorder realizations
      --energy-points ENERGY_POINTS
                            number of points in the spectrum to target
      --h-points H_POINTS   number of disorder strengths to test
      --h-min H_MIN         minimum value of disorder strength h
      --h-max H_MAX         maximum value of disorder strength h
      --nev NEV             number of eigenpairs to compute at each point


## References

<span id="ref1"><sup>1</sup> [Pietracaprina et al., "Shift-invert diagonalization of large many-body localizing spin chains"](https://doi.org/10.21468/SciPostPhys.5.5.045)</span>  
<span id="ref2"><sup>2</sup> [Abanin et al., "Many-body localization, thermalization, and entanglement"](https://doi.org/10.1103/RevModPhys.91.021001)</span>
