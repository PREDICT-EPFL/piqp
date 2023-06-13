---
title: Home
layout: default
nav_order: 1
---

PIQP is an embedded Proximal Interior Point Quadratic Programming solver, which can solve dense and sparse quadratic programs of the form

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& Gx \leq h, \\
& x_{lb} \leq x \leq x_{ub},
\end{aligned}
$$

with primal decision variables $$x \in \mathbb{R}^n$$, matrices $$P\in \mathbb{S}_+^n$$, $$A \in \mathbb{R}^{p \times n}$$,  $$G \in \mathbb{R}^{m \times n}$$, and vectors $$c \in \mathbb{R}^n$$, $$b \in \mathbb{R}^p$$, $$h \in \mathbb{R}^m$$, $$x_{lb} \in \mathbb{R}^n$$, and $$x_{ub} \in \mathbb{R}^n$$. Combining an infeasible interior point method with the proximal method of multipliers, the algorithm can handle ill-conditioned convex QP problems without the need for linear independence of the constraints.

For more detailed technical results see our pre-print:

[**PIQP: A Proximal Interior-Point Quadratic Programming Solver**](https://arxiv.org/abs/2304.00290)<br>
R. Schwan, Y. Jiang, D. Kuhn, C.N. Jones<br>
ArXiv, 2023

### Features

* PIQP is written in header only C++ 14 leveraging the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for vectorized linear algebra.
* Dense and sparse problem formulations are supported. For small dense problems, vectorized instructions and cache locality can be exploited more efficiently.
* Interface to Python with many more to follow.
* Open source under the BSD 2-Clause License.

### Interfaces

PIQP support a wide range of interfaces including
* C/C++ (with Eigen support)
* Python
* Matlab (soon)
* Julia (soon)
* Rust (soon)

### Credits

PIQP is developed by the following people:
* Roland Schwan (main developer)
* Yuning Jiang (methods and maths)
* Daniel Kuhn (methods and maths)
* Colin N. Jones (methods and maths)

All contributors are affiliated with the [Laboratoire d'Automatique](https://www.epfl.ch/labs/la/) and/or the [Risk Analytics and Optimization Chair](https://www.epfl.ch/labs/rao/) at [EPFL](https://www.epfl.ch/), Switzerland.

This work was supported by the [Swiss National Science Foundation](https://www.snf.ch/) under the [NCCR Automation](https://nccr-automation.ch/) (grant agreement 51NF40_180545).

PIQP is build on the following open-source libraries:
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is the work horse under the hood, responsible for producing optimized numerical linear algebra code.
* [ProxSuite](https://github.com/Simple-Robotics/proxsuite) served as an inspiration for the code structure, and the instruction set optimized python bindings. We also utilize some utility functions, and helper macros for cleaner code.
* [SuiteSparse - LDL](https://github.com/DrTimothyAldenDavis/SuiteSparse) (modified version) is for solving linear systems in the sparse solver.
* [pybind11](https://github.com/pybind/pybind11) is used for generating the python bindings.
* [cpu_features](https://github.com/google/cpu_features) is used for run-time instruction set detection in the interface bindings.
* [OSQP](https://github.com/osqp/osqp) served as an inspiration for the C interface.
* [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) served as an inspiration for the iterative refinement scheme.

### License

PIQP is licensed under the BSD 2-Clause License.