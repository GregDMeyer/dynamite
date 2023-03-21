# SYK

## In this example

 - Mapping Majoranas onto spins
 - Imaginary time evolution
 - Ordinary time evolution
 - Generating random states
 - The `Parity` subspace
 - Operators that cross subspaces (in this case, changing the parity)
 - Out-of-time-order correlators
 - Using quantum typicality to compute operators' thermal expectation values
 - Coordinating randomness across MPI ranks

## Overview

This example explores the Sachdev-Ye-Kitaev (SYK) model. In spirit it represents the flipside of the localization explored in the MBL example: the SYK model we explore here is expected to exhibit *fast scrambling*. That is, quantum information is scrambled at the maximum possible rate.

This example also gives us a chance to look at how quantum systems other than spins can be explored with dynamite, by transforming them onto a spin system. The SYK model we'll use consists of Majoranas interacting in 0D, with random couplings. Specifically it consists of every possible 4-body interaction among N Majoranas, with each one having a random coupling strength:

$$H = \frac{6}{N^3} \sum_{ijkl} J_{ijkl} \chi_i \chi_j \chi_k \chi_l$$

where $J_{ijkl}$ are random with some particular distribution (we will use the uniform distribution in the range $[-1, 1]$).

To map the Majoranas onto the spin systems that are natively supported in dynamite, we can use the following transformation. For the Majorana with index $i$, let $q = \lfloor i/2 \rfloor$. Then
$$\chi_i = \sigma^{\{x, y\}}_q \prod_{m \in [0, q-1]} \sigma^z$$
where the first Pauli is $\sigma^x$ if $i$ is even and $\sigma^y$ if it's odd. In words, the Majorana consists of a $\sigma^x$ or $\sigma^y$ with a string of $\sigma^z$ extending to the edge of the spin chain. Note that we get two Majoranas for each spin!

This is straightforward to implement in dynamite, but is actually already built in in the `dynamite.extras` module so we don't have to do it ourselves:


```python
from dynamite.extras import majorana

# a visual look at the Pauli structure of Majorana with index 9
print(majorana(9).table())
```

       coeff. | operator 
    =====================
        1.000 | ZZZZY


In this example project, the Hamiltonian is implemented by `build_hamiltonian` in the file `run_syk.py`. That function uses some clever optimizations to speed things up since our Hamiltonian has so many terms; we also include a more straightforward, but slower implementation called `build_hamiltonian_simple` for comparison. Check out the difference in performance, for a system of only 16 Majoranas (8 spins!):


```python
from run_syk import build_hamiltonian, build_hamiltonian_simple
```


```python
%timeit -n 1 -r 1 build_hamiltonian(N=16)
```

    187 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
%timeit -n 1 -r 1 build_hamiltonian_simple(N=16)
```

    1.98 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


## Goals

In this project we investigate the fast scrambling behavior of the SYK model by studying *out of time order correlators* (OTOCs). In particular, we will measure how much a system is "scrambled" at time $t$ by measuring to what extent two local operators $V(0)$ and $W(t)$ anticommute (where the anticommutator at time $t=0$ is zero). The quantity we will measure is
$$C(t) = \langle \{ W(t), V(0) \}^2 \rangle .$$
It's helpful to reduce this to the following equivalent expression
$$C(t) = 2 \mathrm{Re}\left[ \langle W(t) V(0) W(t) V(0) \rangle \right] + 1/2$$
which is the formulation of $C(t)$ that we will use in the computations here.
(For more details, see the referenced publications).

We are specifically interested in the expectation value of the operator $O(t) = W(t) V(0) W(t) V(0)$ with respect to *thermal states* of various (inverse) temperatures $\beta$. Now, dynamite's speed comes from the fact that it works with pure states, rather than mixed states---so the obvious plan to just compute $\mathrm{Tr} \left[ O(t) e^{-\beta H} \right]$ is out of the question. Instead, we can take advantage of an idea called *quantum typicality* to get an estimate of the expectation value more efficiently (see references below). Quantum typicality says that $\mathrm{Tr} \left[ O(t) e^{-\beta H} \right]$ is approximated by the expectation value of $O(t)$ with respect to random states of temperature $\beta$. 

So, our plan will be to sample a number of random states, use imaginary time evolution to set their temperature to the given value of $\beta$, and then take the expectation value of $O(t)$ with respect to the result. We will also take advantage of the fact that we can rewrite $\mathrm{Tr} \left[ O(t) e^{-\beta H} \right] = \mathrm{Tr} \left[ e^{-\beta H/2} O(t) e^{-\beta H/2} \right]$. For simplicity we can set $W=\chi_0$ and $V=\chi_1$. With that, for a uniformly random state $\left| \psi_r \right>$ we will compute (writing things out in full):

$$\left< \psi_r \right| e^{-\beta H/2} e^{iHt} \chi_0 e^{-iHt} \chi_1 e^{iHt} \chi_0 e^{-iHt} \chi_1 e^{-\beta H/2} \left| \psi_r \right>$$

The function `compute_otoc` starts with $\left| \psi_r \right>$ and just works through the operator from right to left, applying imaginary and real time evolution and multiplying by operators as it goes until it finally reaches the other side, when it takes the inner product to find the expectation value.

## Remark: matrix-free methods

The SYK Hamiltonian has a very large number of terms. Most Hamiltonians we encounter have a number of terms that scale perhaps as $L$ or $L^2$, where $L$ is the number of spins. The SYK model has roughly $(2L)^4$ terms. For example, with only 40 Majoranas there are already over 90,000 terms!


```python
H = build_hamiltonian(40)  # this make take a moment to run
H.nterms
```




    91390



Due to this fact, the memory required to store the matrix for this operator becomes very large:


```python
H.estimate_memory()  # returns an estimate in Gb of the memory required to build the matrix for this operator
```




    105.63354624



Of note here is that `H` is a symbolic representation of the operator; the matrix has not been stored in memory. You don't need 105 Gb of RAM to evaluate the above two cells! One of the most important features of dynamite is the ability to perform computations "matrix-free," such that the matrix is *never* built! Instead, matrix elements are generated on the fly when needed. For historical reasons, these are called "shell" matrices in dynamite. Using them results in a drastic reduction in memory usage:


```python
H.shell = True
H.estimate_memory()
```




    0.00219336



In general, the tradeoff is that performing computations matrix-free can be somewhat slower, due to the extra cost of computing the matrix elements. But this is not always true: for matrix-vector multiplications, the limiting factor is often the memory bandwidth, and having to pull less data from memory can actually speed things up. This is especially true when running things in parallel, if many CPUs are sharing the same memory bus. Ultimately, one should just experiment with different configurations and see which one gives the best combination of speed and memory cost.

## Remark: disorder realizations and parallelism

(also discussed in MBL example)

The SYK Hamiltonian is a case in which getting good data requires disorder averaging---that is, running the computation many times with fresh randomness. Given $N$ CPU cores there are two broad ways one can parallelize that process: (1) running $N$ disorder realizations independently at the same time, each using one core, and (2) using MPI to parallelize one computation across all $N$ cores and then doing each disorder realization in sequence. In this case, (1) will almost always be faster and should be prioritized---while the MPI parallelism in dynamite is highly optimized, there will always be some cost to the communication between cores.

However, there are situations in which using MPI may be preferable, for example if running $N$ independent disorder realizations uses too much memory. Ultimately, the user should experiment with different configurations to determine what gives the best performance. Ideally, in practice one would simply make use of a large cluster of GPUs, running independent disorder realizations on each one.

## Remark: using randomness in dynamite

(also discussed in MBL example)

One needs to take extra care when using randomness in code that will be run under MPI with multiple ranks. Each rank is its own Python process, and by default will have a different random seed---so if you are not careful, each of your ranks may end up trying to build a different Hamiltonian! (dynamite does check that the Hamiltonian is the same on all ranks before building the underlying matrix, so you will get an error if this happens).

In this example, we handle this by explicitly setting the random number generator's seed. This is convenient because it means that the "random" numbers will be the same on each rank. It also makes your science more reproducible. You have to be careful, however, to remember to change that seed when you want new disorder realizations! The example given here has a command line switch to set the seed.

If you want, there is another way to handle this issue: have MPI rank 0 pick a seed, and share it with the rest of the ranks. A code snippet implementing this can be found in the file `bcast_seed.py` in this directory.

**Note 1:** when setting a random state via `State(state='random')`, dynamite is already careful about coordinating randomness between MPI ranks, so the user does not need to worry about it in this case.

**Note 2:** If you will never run your code on multiple MPI ranks, you don't need to worry about this at all. In particular, running on a GPU with 1 CPU process will not encounter this issue.

## Remark: re-using time evolution

An important aspect of dynamite's time evolution algorithm is that its runtime is roughly proportional to $|| Ht ||$. In `run_syk.py`, we take advantage of this to reduce computational costs. Suppose we want to compute $C(t)$ for two temperatures $\beta_1$ and $\beta_2$, with $\beta_1 < \beta_2$. Starting with a random state $\left| \psi_r \right>$, one might compute

$$\left| \psi_{\beta_1} \right> = e^{-\beta_1 H} \left| \psi_r \right>$$
$$\left| \psi_{\beta_2} \right> = e^{-\beta_2 H} \left| \psi_r \right>$$

However, it will be faster to compute
$$\left| \psi_{\beta_1} \right> = e^{-\beta_1 H} \left| \psi_r \right>$$
$$\left| \psi_{\beta_2} \right> = e^{-(\beta_2-\beta_1) H} \left| \psi_{\beta_1} \right>$$
because in this case, the exponent of the second evolution has smaller norm.

Note that if one is computing the OTOC for multiple times $t$, one can take this a step further and do a similar thing for some of the time evolution operators $e^{-i H t}$ in the definition of the OTOC. (This is not implemented in this example).

## Usage

The computation is implemented in `run_syk.py`. The script will output, in CSV format, the value of $C(t)$ for each combination of $\beta$ and $t$ specified. Here are the command line options:


```python
! python run_syk.py -h
```

    usage: run_syk.py [-h] [-N N] [-b B] [-t T] [--H-iters H_ITERS]
                      [--state-iters STATE_ITERS] [-s SEED] [--no-shell]
    
    Compute OTOCs for the SYK model.
    
    options:
      -h, --help            show this help message and exit
      -N N                  number of majoranas
      -b B                  comma-separated list of values of beta
      -t T                  comma-separated list of values of the time t
      --H-iters H_ITERS     number of Hamiltonian disorder realizations
      --state-iters STATE_ITERS
                            number of random states per Hamiltonian
      -s SEED, --seed SEED  seed for the random number generator
      --no-shell            disable shell matrices (they are enabled by default)


Try running this computation with MPI, or on a GPU if you have one, and compare the performance! You can also try disabling shell matrices with the `--no-shell` option to see first-hand how quickly the memory usage blows up for this Hamiltonian.

## References

TODO