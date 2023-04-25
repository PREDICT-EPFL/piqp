---
title: Settings
layout: default
parent: Interfaces
nav_order: 3
---

All interfaces have the same internal solver settings. Note that the default settings have been tuned for 64bit floating point data types (i.e. `double`). If the solver is run with 32bit floating point data types (i.e. `float`) this can result in convergence issues. In this case, the tolerances have to be reduced.

| Argument                  | Default Value | Description                                                         |
|:--------------------------|:--------------|:--------------------------------------------------------------------|
| rho_init                  | 1e-6          | Initial value for the primal proximal penalty parameter rho.        |
| delta_init                | 1e-3          | Initial value for the augmented lagrangian penalty parameter delta. |
| eps_abs                   | 1e-8          | Absolute tolerance.                                                 |
| eps_rel                   | 1e-9          | Relative tolerance.                                                 |
| check_duality_gap         | true          | Check terminal criterion on duality gap.                            |
| eps_duality_gap_abs       | 1e-8          | Absolute tolerance on duality gap.                                  |
| eps_duality_gap_rel       | 1e-9          | Relative tolerance on duality gap.                                  |
| reg_lower_limit           | 1e-10         | Lower limit for regularization.                                     |
| reg_finetune_lower_limit   | 1e-13         | Fine tune lower limit regularization.                               |
| max_iter                  | 200           | Maximum number of iterations.                                       |
| max_factor_retires        | 10            | Maximum number of factorization retires before failure.             |
| preconditioner_scale_cost | false         | Scale cost in Ruiz preconditioner.                                  |
| preconditioner_iter       | 10            | Maximum of preconditioner iterations.                               |
| tau                       | 0.995         | Maximum interior point step length.                                 |
| verbose                   | false         | Verbose printing.                                                   |
| compute_timings           | false         | Measure timing information internally.                              |
