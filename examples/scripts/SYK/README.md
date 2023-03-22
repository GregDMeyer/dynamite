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

This example explores the Sachdev-Ye-Kitaev (SYK) model. In spirit it represents the opposite of the localization explored in the MBL example: it is expected to scramble information at the maximum possible rate.[<sup>1,2</sup>](#ref1) Furthermore it exhibits *maximal chaos*: the Lyapunov exponent, which characterizes how rapidly chaotic trajectories diverge, saturates its upper bound of $2\pi T$, where $T$ is the temperature of the system.[<sup>3</sup>](#ref3) Its physics can be connected to the dynamics of quantum information in black holes, providing a testbed for exotic phenomena such as scrambling-based teleportation.[<sup>4,5,6,7</sup>](#ref4) The example code here mirrors closely a study which used dynamite to show numerical evidence for many-body chaos and gravitational dynamics in the SYK model.[<sup>8</sup>](#ref8)

The SYK model gives us a chance to look at how quantum systems other than spins can be explored with dynamite, by transforming them onto a spin system. The SYK model we'll use consists of Majoranas interacting in 0D, with random couplings. Specifically it consists of every possible 4-body interaction among N Majoranas, with each term having a random coupling strength:

$$H = \sqrt{\frac{6}{N^3}} \sum_{ijkl} J_{ijkl} \chi_i \chi_j \chi_k \chi_l$$
where $J_{ijkl}$ are randomly chosen from a Gaussian distribution with variance 1.

To map the Majoranas onto the spin systems that are natively supported in dynamite, we can use the following transformation. For the Majorana with index $i$, let $q = \lfloor i/2 \rfloor$. Then

$$\chi_i = \sigma^{\lbrace x, y\rbrace}_q \prod\limits_{m \in [0, q-1]} \sigma^z_m$$

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

    180 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
%timeit -n 1 -r 1 build_hamiltonian_simple(N=16)
```

    1.98 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


## Goals

In this project we investigate the fast scrambling behavior of the SYK model by studying *out of time order correlators* (OTOCs). In particular, we will measure to what extent two local operators $V(0)$ and $W(t)$ anticommute for various times $t$, where the anticommutator at time $t=0$ is zero:
$$C(t) = \langle \left| \lbrace W(t), V(0) \rbrace \right| ^2 \rangle .$$
It's helpful to reduce this to the following equivalent expression
$$C(t) = 2 \mathrm{Re}\left[ \langle W(t) V(0) W(t) V(0) \rangle \right] + 1/2$$
which is the formulation of $C(t)$ that we will use in the computations here.
(For more details, see the referenced publications).

We are specifically interested in the expectation value of the operator $O(t) = W(t) V(0) W(t) V(0)$ with respect to *thermal states* of various (inverse) temperatures $\beta$. Now, dynamite's speed comes from the fact that it works with pure states, rather than mixed states---so the obvious plan to just compute $\mathrm{Tr} \left[ O(t) e^{-\beta H} \right]$ is out of the question. Instead, we can take advantage of an idea called *quantum typicality* to get an estimate of the expectation value more efficiently (see references below). Quantum typicality says that $\mathrm{Tr} \left[ O(t) e^{-\beta H} \right]$ is approximated by the expectation value of $O(t)$ with respect to random states of temperature $\beta$. 

So, our plan will be to sample a number of random states, use imaginary time evolution to set their temperature to the given value of $\beta$, and then take the expectation value of $O(t)$ with respect to the result. We will also take advantage of the fact that we can rewrite $\mathrm{Tr} \left[ O(t) e^{-\beta H} \right] = \mathrm{Tr} \left[ e^{-\beta H/2} O(t) e^{-\beta H/2} \right]$. For simplicity we can set $W=\chi_0$ and $V=\chi_1$. With that, for a uniformly random state $\left| \psi_r \right>$ we will compute (writing things out in full):

$$\left< \psi_r \right| e^{-\beta H/2} e^{iHt} \chi_0 e^{-iHt} \chi_1 e^{iHt} \chi_0 e^{-iHt} \chi_1 e^{-\beta H/2} \left| \psi_r \right>$$

The function `compute_otoc` starts with $\left| \psi_r \right>$ and just works through the operator from right to left, applying imaginary and real time evolution and multiplying by operators as it goes until it finally reaches the other side, when it takes the inner product to find the expectation value.

## Remark: matrix-free methods

The SYK Hamiltonian has a very large number of terms. Most Hamiltonians we encounter have a number of terms that scale perhaps as $L$ or $L^2$, where $L$ is the number of spins. This SYK model has roughly $(2L)^4$ terms. For example, with only 40 Majoranas there are already over 90,000 terms!


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

Finally, we note that it is worth thinking holistically about how to best handle difficult numerical problems like this. For example, it has been shown that "sparse SYK," which has many fewer terms, can demonstrate a lot of the same physics with much lower memory and runtime costs.[<sup>9,10</sup>](#ref9)

## Remark: disorder realizations and parallelism

(also discussed in MBL example)

The SYK Hamiltonian is a case in which getting good data requires disorder averaging---that is, running the computation many times with fresh randomness. Given $N$ CPU cores there are two main ways one can parallelize that process: (1) running $N$ disorder realizations independently at the same time, each using one core, and (2) using MPI to parallelize one computation across all $N$ cores and then doing each disorder realization in sequence. In this case, (1) will almost always be faster and should be prioritized---while the MPI parallelism in dynamite is highly optimized, there will always be some cost to the communication between cores.

However, there are situations in which using MPI may be preferable, for example if running $N$ independent disorder realizations uses too much memory. Ultimately, the user should experiment with different configurations to determine what gives the best performance. Ideally, in practice one would simply make use of a large cluster of GPUs, running independent disorder realizations on each one.

## Remark: using randomness in dynamite

(also discussed in MBL example)

One needs to take extra care when using randomness in code that will be run under MPI with multiple ranks. Each rank is its own Python process, and by default will have a different random seed---so if you are not careful, each of your ranks may end up trying to build a different Hamiltonian! (dynamite does check that the Hamiltonian is the same on all ranks before building the underlying matrix, so you will get an error if this happens).

There are two ways to handle this: one is to have rank 0 pick a random seed and use MPI to communicate it to all the other ranks, and the other is to simply pass a seed on the command line. Both are implemented in this example for demonstration purposes: if no seed is passed on the command line, then one is generated and communicated to all MPI ranks. If you pass a random seed on the command line, make sure to change it each time you run the code if you want new disorder realizations!

**Note 1:** when setting a random state via `State(state='random')`, dynamite is already careful about coordinating randomness between MPI ranks, so the user does not need to worry about it in that case.

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
      -s SEED, --seed SEED  seed for random number generator. if omitted, a random
                            seed is chosen by querying system hardware randomness
      --no-shell            disable shell matrices (they are enabled by default)


Try running this computation with MPI, or on a GPU if you have one, and compare the performance! You can also try disabling shell matrices with the `--no-shell` option to see first-hand how quickly the memory usage blows up for this Hamiltonian.

## References

<span id="ref1"><sup>1</sup> [Kitaev, "A simple model of quantum holography"](https://online.kitp.ucsb.edu/online/entangled15/kitaev/)</span>  
<span id="ref2"><sup>2</sup> [Maldacena and Stanford, "Remarks on the Sachdev-Ye-Kitaev model"](https://doi.org/10.1103/PhysRevD.94.106002)</span>  
<span id="ref3"><sup>3</sup> [Maldacena et al., "A bound on chaos"](https://doi.org/10.1007/JHEP08(2016)106)</span>  
<span id="ref4"><sup>4</sup> [Gao et al., "Traversable wormholes via a double trace deformation"](https://doi.org/10.1007/JHEP12(2017)151)</span>  
<span id="ref5"><sup>5</sup> [Maldacena et al., "Diving into traversable wormholes"](https://doi.org/10.1002/prop.201700034)</span>  
<span id="ref6"><sup>6</sup> [Brown et al., "Quantum Gravity in the Lab: Teleportation by Size and Traversable Wormholes"](https://doi.org/10.1103/PRXQuantum.4.010320)</span>  
<span id="ref7"><sup>7</sup> [Schuster et al., "Many-Body Quantum Teleportation via Operator Spreading in the Traversable Wormhole Protocol"](https://doi.org/10.1103/PhysRevX.12.031013)</span>  
<span id="ref8"><sup>8</sup> [Kobrin et al., "Many-Body Chaos in the Sachdev-Ye-Kitaev Model"](https://doi.org/10.1103/PhysRevLett.126.030602)</span>  
<span id="ref9"><sup>9</sup> [Xu et al., "A Sparse Model of Quantum Holography"](https://doi.org/10.48550/arXiv.2008.02303)</span>  
<span id="ref10"><sup>10</sup> [Cáceres et al., "Sparse SYK and traversable wormholes"](https://doi.org/10.1007/JHEP11(2021)015)</span>  
