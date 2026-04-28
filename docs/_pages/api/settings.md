---
title: Settings
layout: default
parent: API
nav_order: 1
---

All interfaces have the same internal solver settings. Note that the default settings have been tuned for 64bit floating point data types (i.e. `double`). If the solver is run with 32bit floating point data types (i.e. `float`) this can result in convergence issues. In this case, the tolerances have to be reduced.

| Argument                                         | Default Value                      | Description                                                                                                                         |
|:-------------------------------------------------|:-----------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| `rho_init`                                       | `1e-6`                             | Initial value for the primal proximal penalty parameter rho.                                                                        |
| `delta_init`                                     | `1e-6`                             | Initial value for the augmented lagrangian penalty parameter delta.                                                                 |
| `rho_eq_factor`                                  | `1e-3`                             | Multiplicative factor applied to `rho_init` for equality-only problems (no inequality or bound constraints).                        |
| `delta_eq_factor`                                | `1e-3`                             | Multiplicative factor applied to `delta_init` for equality-only problems (no inequality or bound constraints).                      |
| `eps_abs`                                        | `1e-8`                             | Absolute tolerance.                                                                                                                 |
| `eps_rel`                                        | `1e-9`                             | Relative tolerance.                                                                                                                 |
| `check_duality_gap`                              | `true`                             | Check terminal criterion on duality gap.                                                                                            |
| `eps_duality_gap_abs`                            | `1e-8`                             | Absolute tolerance on duality gap.                                                                                                  |
| `eps_duality_gap_rel`                            | `1e-9`                             | Relative tolerance on duality gap.                                                                                                  |
| `infeasibility_threshold`                        | `0.9`                              | Threshold value for infeasibility detection.                                                                                        |
| `reg_lower_limit`                                | `1e-10`                            | Lower limit for regularization.                                                                                                     |
| `reg_finetune_lower_limit`                       | `1e-13`                            | Fine tune lower limit regularization.                                                                                               |
| `reg_finetune_primal_update_threshold`           | `7`                                | Threshold of number of no primal updates to transition to fine tune mode.                                                           |
| `reg_finetune_dual_update_threshold`             | `7`                                | Threshold of number of no dual updates to transition to fine tune mode.                                                             |
| `max_iter`                                       | `250`                              | Maximum number of iterations.                                                                                                       |
| `max_factor_retires`                             | `10`                               | Maximum number of factorization retires before failure.                                                                             |
| `preconditioner_scale_cost`                      | `false`                            | Scale cost in Ruiz preconditioner.                                                                                                  |
| `preconditioner_reuse_on_update`                 | `false`                            | Reuse the preconditioner from previous setup/update.                                                                                |
| `preconditioner_iter`                            | `10`                               | Maximum of preconditioner iterations.                                                                                               |
| `tau`                                            | `0.99`                             | Maximum interior point step length.                                                                                                 |
| `kkt_solver`                                     | `dense_cholesky`/<br>`sparse_ldlt` | KKT solver backend. Possible values for the dense solver: `dense_cholesky`<br>sparse solver: `sparse_ldlt`, `sparse_multistage`     |
| `iterative_refinement_always_enabled`            | `false`                            | Always run iterative refinement and not only on factorization failure.                                                              |
| `iterative_refinement_eps_abs`                   | `1e-12`                            | Iterative refinement absolute tolerance.                                                                                            |
| `iterative_refinement_eps_rel`                   | `1e-12`                            | Iterative refinement relative tolerance.                                                                                            |
| `iterative_refinement_max_iter`                  | `10`                               | Maximum number of iterations for iterative refinement.                                                                              |
| `iterative_refinement_min_improvement_rate`      | `5.0`                              | Minimum improvement rate for iterative refinement.                                                                                  |
| `iterative_refinement_static_regularization_eps` | `1e-8`                             | Static regularization for KKT system for iterative refinement.                                                                      |
| `iterative_refinement_static_regularization_rel` | `eps^2`                            | Static regularization w.r.t. the maximum abs diagonal term of KKT system.                                                           |
| `max_init_admm_iter`                             | `5`                                | Maximum number of ADMM iterations during initialization.                                                                            |
| `init_mu_scale`                                  | `0.05`                             | Scaling factor for the initial barrier parameter mu.                                                                                |
| `cold_start_alpha`                               | `0.7`                              | Scaling factor for the initial primal guess in cold start.                                                                          |
| `cold_start_sigma`                               | `3.0`                              | Slack penalty parameter (sigma) used during cold start initialization.                                                              |
| `warm_start_sigma`                               | `100.0`                            | Slack penalty parameter (sigma) used during warm start initialization.                                                              |
| `warm_start`                                     | `false`                            | Warm start the solver from the previous solution on update and re-solve. See [Warm Starting](#warm-starting).                       |
| `verbose`                                        | `false`                            | Verbose printing.                                                                                                                   |
| `compute_timings`                                | `false`                            | Measure timing information internally.                                                                                              |

## Warm Starting

PIQP supports warm starting the solver from a previous or user-provided solution. This can significantly reduce the number of iterations needed to solve a sequence of related problems (e.g., in model predictive control or parametric optimization). The warm starting technique is based on the approach described in [[Chen et al., 2025]](https://arxiv.org/abs/2512.00693).

There are two ways to warm start the solver:

### Automatic Warm Starting

When `warm_start` is set to `true`, the solver automatically uses the solution from the previous solve as a warm start point after calling `update` followed by `solve`. This is the simplest approach for solving a sequence of problems where the data changes incrementally.

### Manual Warm Starting

All interfaces provide a `set_warm_start` method that allows you to manually specify the warm start point. This is useful when you have an approximate solution from an external source.

The method takes the primal variable `x` as a required argument. The dual variables `y`, `z_l`, `z_u`, `z_bl`, `z_bu` are all optional — if not provided, they default to zero and the solver computes the slack variables from `x`.
