---
title: Differentiation
layout: default
nav_order: 4
---

PIQP can differentiate through the solution of the QP

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& h_l \leq Gx \leq h_u, \\
& x_l \leq x \leq x_u,
\end{aligned}
$$

e.g., to use it as a layer in a learning pipeline. Given the gradients of a scalar loss $$L$$ with respect to the solution $$(x, y, z_l, z_u, z_{bl}, z_{bu}, s_l, s_u, s_{bl}, s_{bu})$$, `backward` computes the gradients of $$L$$ with respect to the problem data $$(P, c, A, b, G, h_l, h_u, x_l, x_u)$$ by implicit differentiation of the KKT conditions, as described in [CuPIQP](https://arxiv.org/abs/2610.41021) (see also [Citing PIQP]({{site.baseurl}}/citing)).

## Math

For simplicity, consider the formulation with $$Gx \leq h$$ only. Its solution $$\omega = (x, y, z, s)$$ satisfies the KKT conditions $$F(\omega, \theta) = 0$$ with the problem data $$\theta = (P, c, A, b, G, h)$$ and

$$
F(\omega, \theta) = \begin{bmatrix}
Px + c + A^\top y + G^\top z \\
Ax - b \\
Gx - h + s \\
s \circ z
\end{bmatrix}.
$$

If $$\partial F / \partial \omega$$ is nonsingular at the solution, the implicit function theorem gives the derivative of the solution map

$$
\frac{\partial \omega}{\partial \theta} = -\left(\frac{\partial F}{\partial \omega}\right)^{-1} \frac{\partial F}{\partial \theta}.
$$

Instead of forming this Jacobian, `backward` computes the vector-Jacobian product

$$
\nabla_\theta L = \left(\frac{\partial \omega}{\partial \theta}\right)^\top \nabla_\omega L = \left(\frac{\partial F}{\partial \theta}\right)^\top \lambda, \qquad \left(\frac{\partial F}{\partial \omega}\right)^\top \lambda = -\nabla_\omega L,
$$

i.e., it solves a single linear system for the adjoint variables $$\lambda = (\lambda_x, \lambda_y, \lambda_z, \lambda_s)$$. Since $$\partial F / \partial \omega$$ only differs from the regularized KKT matrix of the forward solve by the regularization, the regularized KKT matrix is factored once at the solution, reusing the symbolic analysis of the forward solve, and the factorization is cached for subsequent `backward()` calls on the same solution. The gradients with respect to the problem data are then

| Data  | Gradient                                             |
|:------|:-----------------------------------------------------|
| $$P$$ | $$\frac{1}{2}(\lambda_x x^\top + x \lambda_x^\top)$$ |
| $$c$$ | $$\lambda_x$$                                        |
| $$A$$ | $$\lambda_y x^\top + y \lambda_x^\top$$              |
| $$b$$ | $$-\lambda_y$$                                       |
| $$G$$ | $$\lambda_z x^\top + z \lambda_x^\top$$              |
| $$h$$ | $$-\lambda_z$$                                       |

The lower bounds $$h_l$$ and the box constraints $$x_l \leq x \leq x_u$$ are handled in the same way with their own slacks and duals, e.g., $$\nabla_{h_l} L = \lambda_{z_l}$$ and $$\nabla_{x_u} L = -\lambda_{z_{bu}}$$.

The sparse gradients have the sparsity patterns of the matrices passed to `setup`, e.g., the gradient with respect to $$P$$ is upper triangular if only the upper triangular part of $$P$$ is passed. Gradients with respect to infinite bounds are zero.

At degenerate solutions, e.g., an active constraint with zero dual, the solution map is not differentiable. In this case, or to get informative gradients across active-set changes, [gradient smoothing](#gradient-smoothing) can be used.

## Usage

Set `enable_backward` before `setup`, then call `backward` after a `solve` which returned `PIQP_SOLVED` or `PIQP_MAX_ITER_REACHED`. Missing gradients are treated as zero. The gradients are valid if the returned status, which is also stored in the gradients, is `PIQP_SOLVED`, and until the next `setup`, `update`, or `backward`.

By default, the gradients are computed at the solution of the last `solve`, reusing its KKT factorization. Optionally, a stored result of an earlier `solve` of the same problem can be passed as `forward_result`.

### Status code

{% root_include _common/backward_status_code_table.md %}

### C++

```c++
piqp::DenseSolver<double> solver;  // or piqp::SparseSolver<double>
solver.settings().enable_backward = true;
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
solver.solve();

// optional arguments grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result
if (solver.backward(dL_dx) == piqp::Status::PIQP_SOLVED) {
    const auto& grads = solver.gradients();  // grads.P, grads.c, ...
}
```

### Python

```python
solver = piqp.DenseSolver()  # or piqp.SparseSolver()
solver.settings.enable_backward = True
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u)
solver.solve()

# keyword arguments grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result
if solver.backward(grad_x=dL_dx) == piqp.PIQP_SOLVED:
    dL_dc = solver.gradients.c  # P, A, G are scipy CSC matrices for the sparse solver
```

The fields of `solver.gradients` are read-only views into the solver memory and are overwritten by the next `backward`. Use `.copy()` to keep them.

### C

```c
settings->enable_backward = 1;
piqp_setup_dense(&work, data, settings);  // or piqp_setup_sparse
piqp_solve(work);

// arguments grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result (NULL if not provided)
if (piqp_backward(work, dL_dx, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == PIQP_SOLVED) {
    const piqp_float* dL_dc = work->gradients->c;
}
```

The dense interface stores `P`, `A`, and `G` row-major, the sparse interface provides `P_csc`, `A_csc`, and `G_csc`.

### Matlab / Octave

```matlab
solver = piqp('dense');  % or piqp('sparse')
solver.update_settings('enable_backward', true);
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
solver.solve();

% name/value pairs or a struct with grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result
grads = solver.backward('grad_x', dL_dx);
if grads.status_val == 1  % PIQP_SOLVED
    dL_dc = grads.c;
end
```

## Gradient Smoothing

With `gradient_smoothing`, `backward` differentiates at the nearby point with $$s \circ z = \mu$$ instead of the solution. This makes the gradients smooth across active-set changes at the cost of a bias, which vanishes as $$\mu \to 0$$.

| Setting                       | Default | Description                                          |
|:------------------------------|:--------|:-----------------------------------------------------|
| `gradient_smoothing`          | `false` | Differentiate at the point with $$s \circ z = \mu$$. |
| `gradient_smoothing_mu`       | `1e-3`  | Smoothing parameter $$\mu$$.                         |
| `gradient_smoothing_max_iter` | `10`    | Maximum number of Newton iterations.                 |

The point is found with Newton iterations, which use the same tolerances as `solve`. Constraints with equal lower and upper bounds are treated as equality constraints and are not smoothed.
