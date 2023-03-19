# Floquet

## Model

In this project we will track the time evolution of various states under a time-dependent Floquet Hamiltonian. This system is physically interesting for a number of reasons, not least of which that it can exhibit [Floquet prethermalization](https://doi.org/10.1103/PhysRevResearch.1.033202), which can support out-of-equilibrium phases of matter like [time crystals](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011043)!

The specific model we will implement is the following. The 1D spin chain will evolve under a long range $ZZ$ interaction decaying as a power law, along with a nearest-neighbor $XX$ interaction and a uniform, static magnetic field $\vec{h}$:
$$H = J_z \sum_{i,j} \frac{\sigma^z_i \sigma^z_j}{|i-j|^\alpha} + J_x \sum_{\langle i, j \rangle} \sigma^x_i + \sum_i \vec{h} \cdot \vec{\sigma}$$
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




$1\left(1\left[\sum_{i=0}^{10}\sigma^z_{i}\sigma^z_{i+1}\right] + 0.42\left[\sum_{i=0}^{9}\sigma^z_{i}\sigma^z_{i+2}\right] + 0.253\left[\sum_{i=0}^{8}\sigma^z_{i}\sigma^z_{i+3}\right] + \cdots\right) + 0.2\left[\sum_{i=0}^{10}\sigma^x_{i}\sigma^x_{i+1}\right] + \sum_{i=0}^{11}\left(0.2\sigma^x_{i} + 0.15\sigma^y_{i} + 0.1\sigma^z_{i}\right)$



and the global pi-pulse is simply a multiplication by the all-Pauli-$X$ string:


```python
from dynamite.operators import sigmax, index_product

pi_pulse = index_product(sigmax())
pi_pulse
```




$\prod_{i=0}^{11}\sigma^x_{i}$



## Goals

The actual numerics here are pretty straightforward: we will evolve an initial state under the above Floquet model, and at every time $T$ we will compute various statistics about the state, such as expectation values of operators and the entanglement entropy. We will also observe the "energy" of the state with respect to the Hamiltonian averaged over a full cycle of length $2T$:
$$D_\mathrm{eff} = J_z \sum_{i,j} \frac{\sigma^z_i \sigma^z_j}{|i-j|^\alpha} + J_x \sum_{\langle i, j \rangle} \sigma^x_i + \sum_i h_x \sigma_x$$
That is, the $h_y$ and $h_z$ terms approximately average to zero. 

Of course, since we are driving the system and there is no dissipation, the temperature will eventually go to infinity. However, we hope to observe that for the right frequency (inverse of the period $T$) we should see a "prethermal plateau" in which the system first thermalizes with respect to the $D_\mathrm{eff}$, but the expectation value $\langle D_\mathrm{eff} \rangle$ is approximately conserved. This is *Floquet prethermalization*---the system thermalizes early with respect to an approximate Hamiltonian, and only much later thermalizes to the infinite temperature state.

## Usage

## Checkpointing

