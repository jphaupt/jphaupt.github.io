---
layout: page
title: "CIPSIpy: Quantum Chemistry in Python"
description: High-accuracy quantum chemistry algorithm implementation with JAX and AI pair-programming
img: assets/img/cipsipy-exponential-wall.png
category: fun
---

In January 2026, I started a postdoc wherein I was faced with quite a lot of new material I needed to master (though none wholly unfamiliar). Two of these topics were JAX and a high-accuracy (multireference) quantum chemistry algorithm called Configuration Interaction using a Perturbative Selection made Iteratively (CIPSI).[^newthings]

[CIPSIpy](https://github.com/jphaupt/CIPSIpy) is a minimal implementation of CIPSI
(Configuration Interaction using a Perturbative Selection made Iteratively) in
[JAX](https://github.com/jax-ml/jax), built as a learning project. My goals were to understand the algorithm and its parallelism, get experience with JAX acceleration and have a working implementation of an advanced quantum chemistry algorithm. I also used the opportunity to try out agentic AI coding tools, since that seems to be all the rage at the moment. I was curious how well an LLM can code up advanced (and somewhat niche) numerical algorithms.

For CIPSI, I heavily relied on [Garniron's PhD thesis](https://zenodo.org/records/2558127), for JAX I used [Deep Learning with JAX](https://www.manning.com/books/deep-learning-with-jax) and I used [GitHub Copilot](https://github.com/features/copilot)[^LLM].

## Quantum Chemistry Primer

Note: you don't need to understand every bit of this section to get the gist of the algorithm.

In essence, quantum chemistry can be summarised as a set of sophisticated methods for solving the so-called electronic Schrödinger equation:[^theoryofeverything]

$$
\hat H \Psi = E\Psi
$$

where $E$ is the energy of the system (e.g. a molecule), $\Psi$ is the wave function and the electronic Hamiltonian operator is

$$
\hat H = -\sum_{i} \frac{1}{2} \nabla_i^2 - \sum_{i,I} \frac{Z_I}{r_{iI}} + \sum_{i\gt j} \frac{1}{r_{ij}}.
$$

The typical wave function theory-based approach to solving this equation is by expanding $\Psi$ in terms of basis functions and orbital occupations (called *configurations*). You may know about something called the Pauli exclusion principle, which states that two electrons may not occupy the same space. We may visualise the configurations in terms of orbital occupations, as below:

{% include figure.liquid path="assets/img/configurations.svg" class="img-fluid rounded z-depth-1" %}

Here $\Phi_0$ is what's called the reference configuration (typically found by an optimisation method such as Hartree-Fock), and the other terms are excitations from that reference configuration. Here the arrows represent electrons of up or down spin (often called alpha and beta electrons) occupying an orbital, represented by a horizontal line. Formally, there is an infinite set of orbitals, but if we truncate this we can enumerate all our possible configurations as above, and we can write our Schrödinger equation as a matrix eigenvalue problem, where the matrix elements are between two configurations

$$
H\Psi = E\Psi
$$

where $\Psi$ can be understood as a vector and $H$ a matrix. We are typically most interested in finding the lowest eigenvalue $E_0$.

One fun consequence of the Pauli exclusion principle is that since only one or zero electrons can occupy a state, we can represent our states in terms of bitstrings. Take for example $\Psi_i^a$ in the diagram. For spin down (beta), the first three orbitals are occupied. For alpha, the first, second, and fifth are occupied. There are six orbitals. Therefore, we can write

$$
\Psi_i^a = (\texttt{010011}, \texttt{000111})
$$

where 1 represents occupied and 0 unoccupied. This is a huge performance boon. Bit-level representation turns combinatorrics into relatively cheap bitwise operations, e.g. counting occupied orbitals is a popcount, checking occupancy is an AND operation, generating an excitations is an XOR.

## The Exponential Wall

So far, so good. Any undergraduate that takes a linear algebra class can diagonalise a matrix, right? Well, the problem is that for a system with $M$ orbitals and $(N_\alpha, N_\beta)$ alpha/beta electrons, the dimension of the matrix (i.e. the number of determinants/configurations) is:

$$
N_{\text{det}} = \binom{M}{N_\alpha}\binom{M}{N_\beta},
$$

which grows combinatorially in $M$. This makes even tiny systems intractible very fast, even on the world's most powerful machines. Therefore, a huge slew of approximations and clever methodologies to solve this equation have been developed. Thus, in addition to the size of the basis set, we have another axis on which to improve, which is the so-called hierarchy of theories.

{% include figure.liquid path="assets/img/hierarchies.svg" class="img-fluid rounded z-depth-1" %}

This ranges from the most basic methods like Hartree-Fock, to perturbative methods like MP2, impressive single-reference methods like coupled cluster to exactly diagonalising the matrix (called FCI). CIPSI is in principle at FCI accuracy, albeit with a lot of clever sampling approximations.

## CIPSI

You can read more about CIPSI in Garniron's thesis, but in a nutshell CIPSI is a sparse matrix sampling method where one CIPSI iteration is four steps:

1. Variational diagonalisation: for the "internal" space, build a small Hamiltonian and diagonalise it exactly and find its lowest eigenpair.
2. External determinant generation: generate every reachable excitation from the internal space -- these are the external determinants.
3. Perturbative selection: for each external determinant $|D_\alpha\rangle$, estimate its
second-order perturbative (PT2) contribution to the energy:

$$
\varepsilon_\alpha^{(2)} = \frac{\left|\sum_{I} c_I \langle D_I | \hat H | D_\alpha \rangle\right|^2}
{E_{\text{var}} - \langle D_\alpha | \hat H | D_\alpha \rangle}.
$$

{: start="4"}
4. Subspace growth: Determinants whose $|\varepsilon_\alpha^{(2)}|$ exceeds a threshold get
added to the internal space, and the cycle repeats. Iterate until the total remaining PT2
correction, $E_{\text{PT2}} = \sum_\alpha \varepsilon_\alpha^{(2)}$, drops below a convergence
tolerance -- at which point $E_{\text{var}} + E_{\text{PT2}}$ is the final energy estimate.

These steps are reflected in my codebase directly, e.g. `determinants.py`, `hamiltonian.py`.

### Avoiding the Exponential Wall

While no FCI-based method can entirely bypass this basic property of the Schrödinger equation, CIPSI does an excellent job of minimising it. Since this algorithm has many moving parts and is quite complicated (and because I was using an LLM agent I didn't entirely trust), I also wrote an extensive test suite. This handily meant I have some examples to provide in this blog post. They are kept small so that they can run on GitHub CI as well as on my personal laptop (which has 4 cores and no real GPU).

**H2** (STO-3G) is the 2-electron "hello world," small enough to check by
hand. **HeH+** (STO-3G) is a cation, useful for catching spin-asymmetry bugs that a
closed-shell-like H2 case can hide. **H3+** (3-21G) is the first system big enough to show
realistic determinant growth without being slow. **Li** (STO-3G) is the one open-shell
(odd-electron) system in the set. **LiH** (6-31G*) is the stress test: 15 spatial orbitals and,
per the exponential-wall formula above, the largest combinatorial space of the five. I ran CIPSI
to full PT2 convergence on all five and compared the FCI determinant count against what
CIPSI actually needed to keep. Below you can see the size of the Hilbert space vs what CIPSI's variational space converged to, per system.

{% include figure.liquid path="assets/img/exponential_wall.png" class="img-fluid rounded z-depth-1" %}

Except for particularly small systems such as HeH+, the saving is quite drastic. This gets even more impressive once you go to the effort to parallelise CIPSI and run on massive clusters to do realistic chemical systems.[^qp]

### Another View of the Selection

TODO: write about the the plot |c_i|^2

## Performance Bottlenecks

TODO: write about Davidson improvement, JAX parallelism, compiling woes, testing on GPUs, etc. and some outlook

TODO: another plot for performance gain

## A Journey in AI Pair-Programming

TODO write about how it made some things waaayyy faster, and the importance of knowing what you are doing

TODO

---

[^newthings]: Actually, the most important topics for this postdoc were quantum Monte Carlo algorithms (DMC, which was somewhat new to me) and deep learning foundation models for atomistic simulations. High-performance Python with JAX and selected CI/CIPSI were secondary, but those are the focus on this particular project.

[^LLM]: I only used GitHub Copilot since I had an academic license so it was free. Since trying other tools, I think this project would have been easier if I used e.g. Claude Code.

[^theoryofeverything]: This equation has even be called [The (everyday) Theory of Everything](https://www.pnas.org/doi/pdf/10.1073/pnas.97.1.28) in a rather inspiring essay by two highly influential physicists Laughlin and Pines.

[^qp]: I did not do this much, but if you are interested in the algorithm you can see its performance in the quantum package software and the associated publications, or look at Garniron's thesis.
