## Problem Formulation

PIQP expects QP of the form

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& Gx \leq h, \\
& x_{lb} \leq x \leq x_{ub}
\end{aligned}
$$

with primal decision variables $$x \in \mathbb{R}^n$$, matrices $$P\in \mathbb{S}_+^n$$, $$A \in \mathbb{R}^{p \times n}$$,  $$G \in \mathbb{R}^{m \times n}$$, and vectors $$c \in \mathbb{R}^n$$, $$b \in \mathbb{R}^p$$, $$h \in \mathbb{R}^m$$, $$x_{lb} \in \mathbb{R}^n$$, and $$x_{ub} \in \mathbb{R}^n$$.

### Example QP

In the following the C++ interface of PIQP will be introduced using the following example QP problem:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top \begin{bmatrix} 6 & 0 \\ 0 & 4 \end{bmatrix} x + \begin{bmatrix} -1 \\ -4 \end{bmatrix}^\top x \\
\text {s.t.}\quad & \begin{bmatrix} 1 & -2 \end{bmatrix} x = 1, \\
& \begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix} x \leq \begin{bmatrix} 0.2 \\ -1 \end{bmatrix}, \\
& -1 \leq x_1 \leq 1.
\end{aligned}
$$