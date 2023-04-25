| Status Code            | Value | Description                                              |
|:-----------------------|:-----:|:---------------------------------------------------------|
| PIQP_SOLVED            |   1   | Solver solved problem up to given tolerance.             |
| PIQP_MAX_ITER_REACHED  |  -1   | Iteration limit was reached.                             |
| PIQP_PRIMAL_INFEASIBLE |  -2   | The problem is primal infeasible.                        |
| PIQP_DUAL_INFEASIBLE   |  -3   | The problem is dual infeasible.                          |
| PIQP_NUMERICS          |  -8   | Numerical error occurred during solving.                 |
| PIQP_UNSOLVED          |  -9   | The problem is unsolved, i.e., `solve` was never called. |
| PIQP_INVALID_SETTINGS  |  -10  | Invalid settings were provided to the solver.            |