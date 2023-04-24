# Floquet

## In this example

 - Initializing product states
 - Time evolution under piecewise Hamiltonians
 - Computing entanglement entropy
 - Computing expectation values
 - Tracking each of the above throughout an evolution
 - Checkpointing and restarting from a checkpoint
 
## Overview

In this project we will track the time evolution of various states under a time-dependent Floquet Hamiltonian. The quantum system we analyze is physically interesting for a number of reasons, not least of which that it can exhibit Floquet prethermalization,[<sup>1</sup>](#ref1) which can support out-of-equilibrium phases of matter like time crystals! [<sup>2</sup>](#ref2)

The specific model we will implement is the following. The 1D spin chain will evolve under a long range $ZZ$ interaction decaying as a power law, along with a nearest-neighbor $XX$ interaction and a uniform, static magnetic field $\vec{h}$:
$$H = J_z \sum_{i<j} \frac{\sigma^z_i \sigma^z_j}{|i-j|^\alpha} + J_x \sum_{\langle i, j \rangle} \sigma^x_i + \sum_i \vec{h} \cdot \vec{\sigma}$$
where the angle brackets on the second term indicates it is only a nearest-neighbor interaction.

In addition, after every period $T$ of time evolution the system will undergo a global $\pi$-pulse, rotating all spins by $180^\circ$ around the $X$-axis. (We can equivalently think of this as flipping the direction of the magnetic field $\vec{h}$ across the $X$ axis every time $T$).

The Hamiltonian $H$ is implemented in `build_hamiltonian` in `run_floquet.py`:


```python
from run_floquet import build_hamiltonian

# for this one its easiest to just globally set L
from dynamite import config
config.L = 12

build_hamiltonian(1.25, 1, 0.2, (0.2, 0.15, 0.1))
```




$\sum\limits_{i=0}^{10}0.25\sigma^z_{i}\sigma^z_{i+1} + 0.42\left[\sum\limits_{i=0}^{9}0.25\sigma^z_{i}\sigma^z_{i+2}\right] + 0.253\left[\sum\limits_{i=0}^{8}0.25\sigma^z_{i}\sigma^z_{i+3}\right] + \cdots + 0.2\left[\sum\limits_{i=0}^{10}0.25\sigma^x_{i}\sigma^x_{i+1}\right] + \sum\limits_{i=0}^{11}\left(0.1\sigma^x_{i} + 0.075\sigma^y_{i} + 0.05\sigma^z_{i}\right)$



and the global pi-pulse is simply a multiplication by the all-Pauli-$X$ string:


```python
from dynamite.operators import sigmax, index_product

pi_pulse = index_product(sigmax())
pi_pulse
```




$\prod\limits_{i=0}^{11}\sigma^x_{i}$



## Goals

The actual numerics here are pretty straightforward: we will evolve an initial state under the above Floquet model, and at every time $T$ we will compute various statistics about the state, such as expectation values of operators and the entanglement entropy. We will also observe the "energy" of the state with respect to the Hamiltonian averaged over a full cycle of length $2T$:
$$D_\mathrm{eff} = J_z \sum_{i,j} \frac{\sigma^z_i \sigma^z_j}{|i-j|^\alpha} + J_x \sum_{\langle i, j \rangle} \sigma^x_i + \sum_i h_x \sigma_x$$
That is, the $h_y$ and $h_z$ terms approximately average to zero. 

Of course, since we are driving the system and there is no dissipation, the temperature will eventually go to infinity. However, we hope to observe that for the right frequency (inverse of the period $T$) we should see a "prethermal plateau" in which $D_\mathrm{eff}$ is approximately conserved, and thus the system can thermalize with respect to it. This is *Floquet prethermalization*---the system thermalizes early with respect to an approximate Hamiltonian, and only much later thermalizes to the infinite temperature state.

## Remark: checkpointing

Especially on HPC clusters, we may encounter situations where our compute jobs are killed before they complete. Or, we may look at the results of a completed job and decide we want to evolve for a longer time. Whatever the reason, the ability to save our progress and re-start computations where we left off can be extremely useful.

This can often be quite easy to accomplish, and it is implemented in this example. Here, every $n$ iterations (where $n$ is set on the command line by the user), the state vector is saved to a file, with the cycle number contained in the filename. When the script starts up, it checks to see if such a file already exists, and if it does, it reads in the vector and starts the evolution from there. Pretty straightforward!

## Remark: operator arithmetic

Note that in `run_floquet.py` we compute the "averaged" Hamiltonian $D_\mathrm{eff}$ simply as

```python
Deff = (H + X*H*X)/2
```
where `X` is the global pi pulse operator.

This showcases the operator arithmetic that is possible in dynamite. You may sometimes find dynamite useful just as a "calculator" for doing arithmetic with strings of Paulis! Note in particular that due to the symbolic way that dynamite stores operators, calculations like the above can be done in milliseconds on a laptop even when multiplying the operators as matrices would be extremely expensive.

## Usage

The computation is implemented in `run_floquet.py`. The script will output, in CSV format, the half-chain entanglement entropy, the effective energy $\langle D_\mathrm{eff} \rangle$, and the expectation value of $S^z$ for each spin. Note that the data is written to stdout and any other information is written to stderr, so you can do for example `python run_floquet.py -L 12 > data.csv` and only the data will be written to the CSV file.

Here are the command line options:


```python
! python run_floquet.py -h
```

    usage: run_floquet.py [-h] [-L L] [--Jx JX] [--h-vec H_VEC] [--alpha ALPHA]
                          [-T T] [--initial-state-dwalls INITIAL_STATE_DWALLS]
                          [--n-cycles N_CYCLES]
                          [--checkpoint-path CHECKPOINT_PATH]
                          [--checkpoint-every CHECKPOINT_EVERY]
    
    Evolve under a Floquet Hamiltonian
    
    options:
      -h, --help            show this help message and exit
      -L L                  number of spins
      --Jx JX               coefficient on the XX term
      --h-vec H_VEC         magnetic field vector
      --alpha ALPHA         power law for long range ZZ interaction
      -T T                  Floquet period
      --initial-state-dwalls INITIAL_STATE_DWALLS
                            Number of domain walls to include in initial product
                            state
      --n-cycles N_CYCLES   Total number of Floquet cycles
      --checkpoint-path CHECKPOINT_PATH
                            where to save the state vector for
                            checkpointing/restarting. [default: ./]
      --checkpoint-every CHECKPOINT_EVERY
                            how frequently to save checkpoints, in number of
                            cycles. if this option is omitted, checkpoints will
                            not be saved.


## References

<span id="ref1"><sup>1</sup> [Machado et al., "Exponentially slow heating in short and long-range interacting Floquet systems"](https://doi.org/10.1103/PhysRevResearch.1.033202)</span>  
<span id="ref2"><sup>2</sup> [Machado et al., "Long-Range Prethermal Phases of Nonequilibrium Matter"](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011043)</span>  
