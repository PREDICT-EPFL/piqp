# PIQP

[![DOI](https://img.shields.io/badge/DOI-10.1109/CDC49753.2023.10383915-green.svg)](https://doi.org/10.1109/CDC49753.2023.10383915)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-blue.svg)](https://arxiv.org/abs/2304.00290)
[![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40__180545)-90e3dc.svg)](https://nccr-automation.ch/)

[![Docs](https://img.shields.io/badge/Docs-available-brightgreen.svg)](https://predict-epfl.github.io/piqp/)
![License](https://img.shields.io/badge/License-BSD--2--Clause-brightgreen.svg)
[![PyPI - downloads](https://img.shields.io/pypi/dm/piqp.svg?label=PyPI%20downloads)](https://pypi.org/project/piqp/)
[![Conda - downloads](https://img.shields.io/conda/dn/conda-forge/piqp.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/piqp)

PIQP is a Proximal Interior Point Quadratic Programming solver, which can solve dense and sparse quadratic programs of the form

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& Gx \leq h, \\
& x_{lb} \leq x \leq x_{ub},
\end{aligned}
$$

Combining an infeasible interior point method with the proximal method of multipliers, the algorithm can handle ill-conditioned convex QP problems without the need for linear independence of the constraints.

## Features

* PIQP is written in header only C++ 14 leveraging the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for vectorized linear algebra.
* Dense and sparse problem formulations are supported. For small dense problems, vectorized instructions and cache locality can be exploited more efficiently.
* Interface to Python with many more to follow.
* Allocation free problem updates and re-solves.
* Open source under the BSD 2-Clause License.

## Interfaces

PIQP support a wide range of interfaces including
* C/C++ (with Eigen support)
* Python
* Matlab/Octave
* R
* Julia (soon)
* Rust (soon)

## Credits

PIQP is developed by the following people:
* Roland Schwan (main developer)
* Yuning Jiang (methods and maths)
* Daniel Kuhn (methods and maths)
* Colin N. Jones (methods and maths)

All contributors are affiliated with the [Laboratoire d'Automatique](https://www.epfl.ch/labs/la/) and/or the [Risk Analytics and Optimization Chair](https://www.epfl.ch/labs/rao/) at [EPFL](https://www.epfl.ch/), Switzerland.

This work was supported by the [Swiss National Science Foundation](https://www.snf.ch/) under the [NCCR Automation](https://nccr-automation.ch/) (grant agreement 51NF40_180545).

PIQP is an adapted implementation of [work](https://link.springer.com/article/10.1007/s10589-020-00240-9) by Spyridon Pougkakiotis and Jacek Gondzio, and is built on the following open-source libraries:
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): It's the work horse under the hood, responsible for producing optimized numerical linear algebra code.
* [ProxSuite](https://github.com/Simple-Robotics/proxsuite): The code structure (folder/namespace structure, etc.), some utility functions/helper macros, and the instruction set optimized python bindings are based on ProxSuite.
* [SuiteSparse - LDL](https://github.com/DrTimothyAldenDavis/SuiteSparse) (modified version): Used for solving linear systems in the sparse solver.
* [pybind11](https://github.com/pybind/pybind11): Used for generating the python bindings.
* [cpu_features](https://github.com/google/cpu_features): Used for run-time instruction set detection in the interface bindings.
* [OSQP](https://github.com/osqp/osqp): The C and Matlab interface is inspired by OSQP.
* [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs): Parts of the iterative refinement scheme are inspired by Clarabel's implementation.

## Citing our Work

If you found PIQP useful in your scientific work, we encourage you to cite our accompanying paper:
```
@INPROCEEDINGS{schwan2023piqp,
  author={Schwan, Roland and Jiang, Yuning and Kuhn, Daniel and Jones, Colin N.},
  booktitle={2023 62nd IEEE Conference on Decision and Control (CDC)}, 
  title={{PIQP}: A Proximal Interior-Point Quadratic Programming Solver}, 
  year={2023},
  volume={},
  number={},
  pages={1088-1093},
  doi={10.1109/CDC49753.2023.10383915}
}
```
The benchmarks are available in the following repo: [piqp_benchmarks](https://github.com/PREDICT-EPFL/piqp_benchmarks)

## License

PIQP is licensed under the BSD 2-Clause License.