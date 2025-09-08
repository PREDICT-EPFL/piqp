---
title: Result
layout: default
parent: API
nav_order: 2
---

| Field                  | Description                                   |
|:-----------------------|:----------------------------------------------|
| `x`                    | Primal solution                               |
| `y`                    | Dual solution of equality constraints         |
| `z_l`                  | Dual solution of lower inequality constraints |
| `z_u`                  | Dual solution of upper inequality constraints |
| `z_bl`                 | Dual solution of lower bound box constraints  |
| `z_bu`                 | Dual solution of upper bound box constraints  |
| `info.status`          | Solver status                                 |
| `info.iter`            | Number of iterations                          |
| `info.primal_obj`      | Primal objective value                        |
| `info.dual_obj`        | Dual objective value                          |
| `info.duality_gap`     | Duality gap                                   |
| `info.duality_gap_rel` | Relative duality gap                          |
| `info.setup_time`      | Setup time                                    |
| `info.update_time`     | Update time                                   |
| `info.solve_time`      | Solve time                                    |
| `info.kkt_factor_time` | KKT factor time                               |
| `info.kkt_solve_time`  | KKT solve time                                |
| `info.run_time`        | Total run time: setup/update + solve          |

{: .warning }
Timing information `info.xxx_time` is only measured if `compute_timings` is set to `true` in the settings.
