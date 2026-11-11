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

e.g., to use it as a layer in a learning pipeline. Given the gradients of a scalar loss $$L$$ with respect to the solution $$(x, y, z_l, z_u, z_{bl}, z_{bu}, s_l, s_u, s_{bl}, s_{bu})$$, the `backward` function computes the gradients of $$L$$ with respect to the problem data $$(P, c, A, b, G, h_l, h_u, x_l, x_u)$$, i.e., a vector-Jacobian product of the solution map. It is computed by implicit differentiation of the KKT conditions at the solution, which requires one solve of the KKT system with the factorization of the forward solve.

## Math

Let $$K$$ be the Jacobian of the KKT conditions with respect to the primal-dual variables at the solution, and $$g$$ the stacked gradients of $$L$$ with respect to the solution. `backward` solves the adjoint system $$K^\top \lambda = -g$$ and returns

| Data    | Gradient                                                                |
|:--------|:------------------------------------------------------------------------|
| $$P$$   | $$\frac{1}{2}(\lambda_x x^\top + x \lambda_x^\top)$$                    |
| $$c$$   | $$\lambda_x$$                                                           |
| $$A$$   | $$\lambda_y x^\top + y \lambda_x^\top$$                                 |
| $$b$$   | $$-\lambda_y$$                                                          |
| $$G$$   | $$(\lambda_{z_u} - \lambda_{z_l}) x^\top + (z_u - z_l) \lambda_x^\top$$ |
| $$h_l$$ | $$\lambda_{z_l}$$                                                       |
| $$h_u$$ | $$-\lambda_{z_u}$$                                                      |
| $$x_l$$ | $$\lambda_{z_{bl}}$$                                                    |
| $$x_u$$ | $$-\lambda_{z_{bu}}$$                                                   |

* The gradient with respect to $$P$$ is the symmetric matrix $$\frac{1}{2}(\lambda_x x^\top + x \lambda_x^\top)$$, i.e., the gradient with respect to $$P$$ as a symmetric matrix, as in, e.g., ProxQP, qpth, and OSQP. The dense solver returns both triangles.
* The gradients of the sparse solver have the sparsity patterns of the matrices passed to `setup`. In particular, if only the upper triangular part of $$P$$ is passed, the gradient with respect to $$P$$ is upper triangular, and its off-diagonal entries are half of the derivative with respect to the corresponding passed entry, since that entry also sets the lower triangular part of $$P$$.
* The gradients with respect to infinite bounds, and with respect to the rows of $$G$$ whose bounds are both infinite, are zero. Gradients with respect to infinite slacks are ignored.
* The gradients with respect to $$z_{bl}, z_{bu}, s_{bl}, s_{bu}$$ have dimension $$n$$, as in the [result]({{site.baseurl}}/api/result).

The solution map is differentiable if the solution is strictly complementary (every active constraint has a strictly positive dual) and satisfies LICQ (the equality and active constraint rows are linearly independent). At degenerate points, e.g., at a constraint which is active with zero dual, the solution map has a kink, and `backward` returns the derivative of the regularized linearization at the returned solution. In this case, or if informative gradients across active-set changes are needed, [gradient smoothing](#gradient-smoothing) can be used.

## Usage

1. Set `enable_backward` **before** `setup`. This allocates the storage for the gradients, such that `backward` does not allocate.
2. Call `setup` (or `update`) and `solve`. The solve has to return `PIQP_SOLVED` or `PIQP_MAX_ITER_REACHED`.
3. Call `backward` with the gradients of the loss with respect to the solution. Missing gradients are treated as zero.
4. If `backward` returns `PIQP_SOLVED`, the gradients with respect to the problem data are available in the gradients object.

The gradients are valid until the next `setup`, `update`, or `backward` call, and a failed `backward` invalidates them as well. The status of the gradients object is the status returned by the last `backward`, and is reset to `PIQP_UNSOLVED` by `setup` and `update`, i.e., the gradients are valid if and only if it is `PIQP_SOLVED`. `backward` does not change the result of the solve, so `backward` can be called multiple times with different gradients after one `solve`.

The argument `forward_result` is optional and selects the point at which the gradients are computed:

* **Not provided (default):** the gradients are computed at the solution of the last `solve`, and, unless gradient smoothing is enabled, the KKT factorization of that `solve` is reused, i.e., `backward` only solves the KKT system. The factorization belongs to the second-to-last iterate, which is close to the solution after convergence.
* **Provided:** the gradients are computed at the stored result of an earlier `solve` of the same problem data, e.g., to differentiate the results of a rollout after all problems have been solved. The KKT system is factorized once at that point. This overwrites the factorization of the last `solve`, so later calls without `forward_result` factorize as well.

With [gradient smoothing](#gradient-smoothing), the factorization of the last `solve` is never reused: the KKT system is always factorized, once per smoothing iteration, with or without `forward_result`.

### Status code

{% root_include _common/backward_status_code_table.md %}

### C++

```c++
piqp::DenseSolver<double> solver;  // or piqp::SparseSolver<double>
solver.settings().enable_backward = true;
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
solver.solve();

// gradients w.r.t. x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu (optional, nullopt is zero), and forward_result
piqp::Status status = solver.backward(dL_dx);
if (status == piqp::Status::PIQP_SOLVED) {
    const auto& grads = solver.gradients();  // grads.status == status
    // grads.P, grads.c, grads.A, grads.b, grads.G, grads.h_l, grads.h_u, grads.x_l, grads.x_u
}
```

### Python

```python
solver = piqp.DenseSolver()  # or piqp.SparseSolver()
solver.settings.enable_backward = True
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u)
solver.solve()

# keyword arguments grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result
status = solver.backward(grad_x=dL_dx)
if status == piqp.PIQP_SOLVED:  # also available as solver.gradients.status
    dL_dc = solver.gradients.c  # P, c, A, b, G, h_l, h_u, x_l, x_u (scipy CSC matrices for the sparse solver)
```

The fields of `solver.gradients` are read-only views into the memory of the solver, i.e., no data is copied, also not for the scipy CSC matrices of the sparse solver. They reflect the results of later `backward` calls, and stay valid across `setup` as long as the problem dimensions (dense) or the sparsity patterns of `P`, `A`, and `G` (sparse) do not change. Otherwise, or to keep the gradients of a `backward` call, copy them, e.g., `solver.gradients.c.copy()`.

A result for `forward_result` has to be copied, since `solver.result` refers to the internal memory of the solver:

```python
result = piqp.Result()
for field in ["x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"]:
    setattr(result, field, np.array(getattr(solver.result, field)))
result.info = solver.result.info
```

### C

```c
settings->enable_backward = 1;
piqp_setup_dense(&work, data, settings);  // or piqp_setup_sparse
piqp_solve(work);

// gradients w.r.t. x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu (NULL is zero), and forward_result (NULL: last solve)
piqp_status status = piqp_backward(work, dL_dx, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
if (status == PIQP_SOLVED) {  // also available as work->gradients->status
    const piqp_float* dL_dc = work->gradients->c;
}
```

For the dense interface, `work->gradients->P`, `A`, and `G` are row-major arrays, for the sparse interface `P_csc`, `A_csc`, and `G_csc` are CSC matrices. The memory is owned by the workspace.

### Matlab / Octave

```matlab
solver = piqp('dense');  % or piqp('sparse')
solver.update_settings('enable_backward', true);
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
result = solver.solve();

% name/value pairs or a struct with the fields grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result
grads = solver.backward('grad_x', dL_dx);
if grads.status_val == 1  % PIQP_SOLVED
    dL_dc = grads.c;  % P, c, A, b, G, h_l, h_u, x_l, x_u
end
```

A failed `backward` does not raise an error, `grads.status` and `grads.status_val` contain the status and the gradient fields are empty.

## Gradient Smoothing

With `gradient_smoothing` enabled, `backward` differentiates at a relaxed point instead of the solution: starting from the solution, whose complementarity $$s \circ z$$ is of the order of the solver tolerance, Newton iterations find the nearby point satisfying the KKT conditions with the larger complementarity $$s \circ z = \mu$$, where $$\mu$$ is `gradient_smoothing_mu`. This makes the gradients smooth across active-set changes, i.e., constraints close to becoming active or inactive contribute to the gradient, at the cost of a bias which vanishes as $$\mu \to 0$$.

| Setting                       | Default | Description                                                         |
|:------------------------------|:--------|:--------------------------------------------------------------------|
| `gradient_smoothing`          | `false` | Differentiate at the relaxed point with $$s \circ z = \mu$$.        |
| `gradient_smoothing_mu`       | `1e-3`  | Relaxation parameter $$\mu$$.                                       |
| `gradient_smoothing_max_iter` | `10`    | Maximum number of Newton iterations.                                |

The Newton iterations use the same termination criterion as `solve`, i.e., `eps_abs` and `eps_rel` on the primal and dual residuals, and in addition on the deviation of $$s \circ z$$ from $$\mu$$. If they do not converge, `backward` returns `PIQP_MAX_ITER_REACHED`.

Constraints with equal lower and upper bounds, i.e., equality constraints given as inequalities (`h_l[i] == h_u[i]` or `x_l[i] == x_u[i]`), have no interior and are not relaxed. Equality constraints which are only implied by several inequality constraints cannot be detected, in that case the smoothing does not converge.
