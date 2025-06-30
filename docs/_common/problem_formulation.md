## Problem Formulation

PIQP expects QP of the form

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& h_l \leq Gx \leq h_u, \\
& x_l \leq x \leq x_u,
\end{aligned}
$$

with primal decision variables $$x \in \mathbb{R}^n$$, matrices $$P\in \mathbb{S}_+^n$$, $$A \in \mathbb{R}^{p \times n}$$,  $$G \in \mathbb{R}^{m \times n}$$, and vectors $$c \in \mathbb{R}^n$$, $$b \in \mathbb{R}^p$$, $$h_l \in \mathbb{R}^m$$, $$h_u \in \mathbb{R}^m$$, $$x_l \in \mathbb{R}^n$$, and $$x_u \in \mathbb{R}^n$$.

{: .note }
PIQP can handle infinite box constraints well, i.e. when elements of $$x_{lb}$$ or $$x_{ub}$$ are $$-\infty$$ or $$\infty$$, respectively. On the contrary, infinite values in the general inequalities $$-\infty = h_l \leq Gx \leq h_u = \infty$$ can cause problems when both sides are unbounded. PIQP internally disables them by setting the corresponding rows in $$G$$ to zero (sparsity structure is preserved). For best performance, consider removing the corresponding constraints from the problem formulation directly.

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