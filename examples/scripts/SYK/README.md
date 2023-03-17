# SYK

This example explores the Sachdev-Ye-Kitaev (SYK) model. In spirit it represents the flipside of the localization explored in the MBL example: the SYK model we explore here is expected to exhibit *fast scrambling*. That is, quantum information is scrambled at the maximum possible rate.

This example also gives us a chance to look at how quantum systems other than spins can be explored with dynamite, by transforming them onto a spin system. The SYK model we'll use consists of Majoranas interacting in 0D, with random couplings. Specifically it consists of every possible 4-body interaction among N Majoranas, with each one having a random coupling strength:

$$H = \sum_{ijkl} J_{ijkl} \chi_i \chi_j \chi_k \chi_l$$

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

    190 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
%timeit -n 1 -r 1 build_hamiltonian_simple(N=16)
```

    1.95 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


## Goals

In this project we investigate the fast scrambling behavior of the SYK model by studying *out of time order correlators* (OTOCs). In particular we will look at the expectation value of the four point correlator $O(t)$:
$$O(t) = \langle W(t) V(0) W(t) V(0) \rangle$$
where $W$ and $V$ are local operators that commute at time $0$.
The intuition is that these correlators measure how much a system is "scrambled" at a particular time $t$ by in effect measuring to what extent $V(0)$ and $W(t)$ commute. (For more details, see the referenced publications).

We are specifically interested in the expectation value of this operator with respect to *thermal states* of various (inverse) temperatures $\beta$. Now, dynamite's speed comes from the fact that it works with pure states, rather than mixed states---so the obvious plan to just compute $\mathrm{Tr} \left[ O(t) e^{\beta H} \right]$ is out of the question. Instead, we can take advantage of an idea called *quantum typicality* to get an estimate of the expectation value more efficiently. TODO: add reference? Quantum typicality says that $\mathrm{Tr} \left[ O(t) e^{\beta H} \right]$ is approximated by the expectation value of $O(t)$ with respect to random states of temperature $\beta$. 

So, our plan will be to sample a number of random states, use imaginary time evolution to set their temperature to the given value of $\beta$, and then take the expectation value of $O(t)$ with respect to the result. We will also take advantage of the fact that we can rewrite $\mathrm{Tr} \left[ O(t) e^{\beta H} \right] = \mathrm{Tr} \left[ e^{\beta H/2} O(t) e^{\beta H/2} \right]$. For simplicity we can set $W=\chi_0$ and $V=\chi_1$. With that, for a uniformly random state $\left| \psi_r \right>$ we will compute (writing things out in full):

$$\left< \psi_r \right| e^{\beta H/2} e^{iHt} \chi_0 e^{-iHt} \chi_1 e^{iHt} \chi_0 e^{-iHt} \chi_1 e^{\beta H/2} \left| \psi_r \right>$$

The function `compute_otoc` starts with $\left| \psi_r \right>$ and just works through the operator from right to left, applying imaginary and real time evolution and multiplying by operators as it goes until it finally reaches the other side, when it takes the inner product to find the expectation value.

## Remark: matrix-free methods

The SYK Hamiltonian has a very large number of terms. Most Hamiltonians we encounter have a number of terms that scale as perhaps $L$ or $L^2$, where $L$ is the number of spins. The SYK model has roughly $(2L)^4$ terms. For example, with only 40 Majoranas there are already over 90,000 terms!


```python
H = build_hamiltonian(40)  # this make take a moment to run
H.msc_size
```




    91390



Due to this fact, the memory required to store the matrix for this operator becomes very large:


```python
H.estimate_memory()  # returns an estimate in Gb
```




    105.63354624



One of the most important features of dynamite is the ability to perform computations "matrix-free," where matrix elements are generated on the fly when needed instead of being stored. For historical reasons, these are called "shell" matrices in dynamite. Using them results in a drastic reduction in memory usage:


```python
H.shell = True
H.estimate_memory()
```




    0.00219336



In general, the tradeoff is that performing computations matrix-free can be somewhat slower, due to the extra cost of computing the matrix elements. But this is not always true: for matrix-vector multiplications, the limiting factor is often the memory bandwidth, and having to pull less data from memory can actually speed things up. This is especially true when running things in parallel, if many CPUs are sharing the same memory bus. Ultimately, one should just experiment with different configurations and see which one gives the best combination of speed and memory cost. # TODO: give example?


```python

```
