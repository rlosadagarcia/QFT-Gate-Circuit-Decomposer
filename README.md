# Gate Circuit Decomposer Project
Attempt at an approximated circuit decomposition of the $\operatorname{QFT}(n)$ (Quantum Fourier Transform operator for $n$ qubits) with less than $n^2$ controlled gates and keeping a low depth. The general quantum gate decomposition method adapted and studied in this project targeting the  $\operatorname{QFT}$ operator is described in [1] and [2]. More general information regarding the code implemented by the authors can be found [in their github repository](https://github.com/rakytap/sequential-quantum-gate-decomposer).
 
[[1]](https://arxiv.org/abs/2109.06770) Péter Rakyta, Zoltán Zimborás, Approaching the theoretical limit in quantum gate decomposition.

[[2]](https://arxiv.org/abs/2203.04426) Péter Rakyta, Zoltán Zimborás, Efficient quantum gate decomposition via adaptive circuit compression.
 
## Objectives achieved in the project:
Succesfully aproximated QFT operator for 2 and 3 qubits with both high gate fidelity and very low tolerance error (up to $\mathcal{O}(10^{-16})$ ) in less than $n^2$ controlled gates and keeping a low depth.

The results are presented in a report (translation pending) which describes the entire project in detail and [can be found in this repository.](https://github.com/rlosadagarcia/QFT-Gate-Circuit-Decomposer/blob/aa02985932aaa03119a3c1d2c961866815bed267/Memoria_Roberto_Losada_Garcia.pdf).
