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

{: .note }
If the problem is detected to be infeasible, the result contains a certificate of infeasibility, normalized to an infinity norm of one.
If `info.status` is `PIQP_PRIMAL_INFEASIBLE`, the dual variables `y`, `z_l`, `z_u`, `z_bl`, and `z_bu` contain a certificate of primal infeasibility, i.e.,
$$A^\top y + G^\top (z_u - z_l) - z_{bl} + z_{bu} = 0$$, $$z_l, z_u, z_{bl}, z_{bu} \geq 0$$, and $$b^\top y - h_l^\top z_l + h_u^\top z_u - x_l^\top z_{bl} + x_u^\top z_{bu} < 0$$, where only finite bounds are taken into account.
If `info.status` is `PIQP_DUAL_INFEASIBLE`, the primal variable `x` contains a certificate of dual infeasibility, i.e.,
$$P x = 0$$, $$A x = 0$$, $$c^\top x < 0$$, and $$x$$ lies in the recession cone of the inequality and bound constraints.
The remaining variables are not meaningful in both cases.
