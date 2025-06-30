---
title: Multistage Problems
layout: default
nav_order: 2
---

PIQP has a specialized backend to solve multistage optimization problems of the following form considerably more efficient:

$$
\begin{aligned}
\min_{x, g} \quad & \sum_{i=0}^{N-1} \ell_i(x_i,x_{i+1},g) + \ell_N(x_N,g) \\
\text {s.t.}\quad & A_ix_i+B_ix_{i+1}+E_ig = b_i,\hspace{0.5em} i=0,\dots,N-1, \\
& C_ix_i+D_ix_{i+1}+F_ig \leq h_i,\hspace{0.40em} i=0,\dots,N-1, \\
& A_Nx_N + E_Ng = b_N, \\
& D_Nx_N + F_Ng \leq h_N,
\end{aligned}
$$

with $$N$$ stages and coupled stage cost

$$
\ell_i \coloneqq \frac{1}{2}\begin{bmatrix}
x_i \\
x_{i+1} \\
g
\end{bmatrix}^\top\begin{bmatrix}
Q_i & S_i^\top & T_i^\top \\
S_i & 0 & 0 \\
T_i & 0 & 0
\end{bmatrix}\begin{bmatrix}
x_i \\
x_{i+1} \\
g
\end{bmatrix}+ c_i^\top x_i,
$$

and terminal cost

$$
\ell_N \coloneqq \frac{1}{2}\begin{bmatrix}
x_N \\
g
\end{bmatrix}^\top\begin{bmatrix}
Q_N & T_N^\top \\
T_N & Q_g
\end{bmatrix}\begin{bmatrix}
x_N \\
g
\end{bmatrix}+ c_N^\top x_N + c_g^\top g,
$$

where $$x_i\in\mathbb{R}^{n_i}$$ are the stage variables, $$g\in\mathbb{R}^{n_g}$$ is a global variable, and $$N \in \mathbb{N}$$ is the horizon. The matrices $$Q_i\in\mathbb{S}_+^{n_i}$$, $$S_i\in\mathbb{R}_+^{n_{i+1}\times n_i}$$, and $$T_i\in\mathbb{R}_+^{n_g\times n_i}$$ together with $$c_i\in\mathbb{R}^{n_i}$$ and $$c_g\in\mathbb{R}^{n_g}$$ form the coupled cost. Stages are also coupled through equality and inequality constraints with $$A_i \in \mathbb{R}^{p_i\times n_i}$$, $$B_i \in \mathbb{R}^{p_i\times n_{i+1}}$$, $$E_i \in \mathbb{R}^{p_i\times n_g}$$, $$b_i \in \mathbb{R}^{p}$$, $$C_i \in \mathbb{R}^{m_i\times n_i}$$, $$D_i \in \mathbb{R}^{m_i\times n_{i+1}}$$, and $$h_i \in \mathbb{R}^{m_i\times n_g}$$, respectively.

An extensive example showcasing the multistage capabilities is the [Robust Scenario MPC Example]({{site.baseurl}}/examples/scenario_example).

## Interface

The interface is the same as for the general sparse QP problem

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& h_l \leq Gx \leq h_u, \\
& x_l \leq x \leq x_u,
\end{aligned}
$$

PIQP detects the structure automatically and extracts the dense blocks from the sparse problem data.

{: .note }
The order of the decision variables is important and has to match the multistage optimization problem, i.e. $$x = (x_0, x_1, \dots, x_N, g)$$. The order of the constraints $$Ax=b$$ and $$h_l \leq Gx \leq h_u$$ can be arbitrary, i.e., gets correctly permuted internally.

Changing the KKT solver backend amounts to adding one line, by changing the `kkt_solver` setting. Make sure you use the sparse solver interface and set the setting **before** the `setup` function.

### C++

```c++
solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage;
```

### C

```c
settings->kkt_solver = PIQP_SPARSE_MULTISTAGE;
```

### Python

```python
solver.settings.kkt_solver = piqp.KKTSolver.sparse_multistage
```

### Matlab

```
solver.update_settings('kkt_solver', 'sparse_multistage');
```
