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
| `delta_init`                                     | `1e-4`                             | Initial value for the augmented lagrangian penalty parameter delta.                                                                 |
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
| `verbose`                                        | `false`                            | Verbose printing.                                                                                                                   |
| `compute_timings`                                | `false`                            | Measure timing information internally.                                                                                              |
