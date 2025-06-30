---
title: Home
layout: default
nav_order: 1
---

PIQP is a Proximal Interior Point Quadratic Programming solver, which can solve dense and sparse quadratic programs of the form

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& h_l \leq Gx \leq h_u, \\
& x_l \leq x \leq x_u,
\end{aligned}
$$

with primal decision variables $$x \in \mathbb{R}^n$$, matrices $$P\in \mathbb{S}_+^n$$, $$A \in \mathbb{R}^{p \times n}$$,  $$G \in \mathbb{R}^{m \times n}$$, and vectors $$c \in \mathbb{R}^n$$, $$b \in \mathbb{R}^p$$, $$h_l \in \mathbb{R}^m$$, $$h_u \in \mathbb{R}^m$$, $$x_l \in \mathbb{R}^n$$, and $$x_u \in \mathbb{R}^n$$. Combining an infeasible interior point method with the proximal method of multipliers, the algorithm can handle ill-conditioned convex QP problems without the need for linear independence of the constraints.

For more detailed technical results see our papers:

[**PIQP: A Proximal Interior-Point Quadratic Programming Solver**](https://ieeexplore.ieee.org/document/10383915)<br>
R. Schwan, Y. Jiang, D. Kuhn, C.N. Jones<br>
IEEE Conference on Decision and Control (CDC), 2023

[**Exploiting Multistage Optimization Structure in Proximal Solvers**](https://arxiv.org/abs/2503.12664)<br>
R. Schwan, D. Kuhn, C.N. Jones<br>
ArXiv, 2025

### Features

* PIQP is written in header only C++ 14 leveraging the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for vectorized linear algebra.
* Dense and sparse problem formulations are supported. For small dense problems, vectorized instructions and cache locality can be exploited more efficiently.
* Special backend for multistage optimization problems.
* Allocation free problem updates and re-solves.
* Open source under the BSD 2-Clause License.

### Interfaces

PIQP support a wide range of interfaces including
* C/C++ (with Eigen support)
* Python
* Matlab/Octave
* R

### Credits

PIQP is developed by the following people:
* Roland Schwan (main developer)
* Yuning Jiang (methods and maths)
* Daniel Kuhn (methods and maths)
* Colin N. Jones (methods and maths)

All contributors are affiliated with the [Laboratoire d'Automatique](https://www.epfl.ch/labs/la/) and/or the [Risk Analytics and Optimization Chair](https://www.epfl.ch/labs/rao/) at [EPFL](https://www.epfl.ch/), Switzerland.

This work was supported by the [Swiss National Science Foundation](https://www.snf.ch/) under the [NCCR Automation](https://nccr-automation.ch/) (grant agreement 51NF40_180545).

PIQP is an adapted implementation of [work](https://link.springer.com/article/10.1007/s10589-020-00240-9) by Spyridon Pougkakiotis and Jacek Gondzio, and is built on the following open-source libraries:
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): It's the work horse under the hood, responsible for producing optimized numerical linear algebra code.
* [Blasfeo](https://github.com/giaf/blasfeo): Used in the sparse_multistage KKT solver backend.
* [ProxSuite](https://github.com/Simple-Robotics/proxsuite): The code structure (folder/namespace structure, etc.), some utility functions/helper macros, and the instruction set optimized python bindings are based on ProxSuite.
* [SuiteSparse - LDL](https://github.com/DrTimothyAldenDavis/SuiteSparse) (modified version): Used for solving linear systems in the sparse solver.
* [pybind11](https://github.com/pybind/pybind11): Used for generating the python bindings.
* [cpu_features](https://github.com/google/cpu_features): Used for run-time instruction set detection in the interface bindings.
* [OSQP](https://github.com/osqp/osqp): The C and Matlab interface is inspired by OSQP.
* [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs): Parts of the iterative refinement scheme are inspired by Clarabel's implementation.

### License

PIQP is licensed under the BSD 2-Clause License.