# Gate Circuit Decomposer Project
Attempt at an approximated circuit decomposition of the $\operatorname{QFT}(n)$ (Quantum Fourier Transform operator for $n$ qubits) with less than $n^2$ controlled gates and keeping a low depth. The general quantum gate decomposition method adapted for this case is described in [[1]](https://arxiv.org/abs/2109.06770) and  [[2]](https://arxiv.org/abs/2203.04426); general info about their code is [in their github repository](https://github.com/rakytap/sequential-quantum-gate-decomposer).
 
## Actual accomplishments:
Succesfully aproximated QFT operator for 2 and 3 qubis with both high gate fidelity and very low tolerance error (up to $\mathcal{O}(10^{-16})$ ) in less than $n^2$ controlled gates and keeping a low depth.
