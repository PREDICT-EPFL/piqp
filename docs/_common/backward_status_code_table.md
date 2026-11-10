`backward` returns the same status codes as `solve`, with the following meaning:

| Status Code           | Value | Description                                                                                                                                                                                                   |
|:----------------------|:-----:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PIQP_SOLVED           |   1   | Gradients computed successfully.                                                                                                                                                                              |
| PIQP_MAX_ITER_REACHED |  -1   | Gradient smoothing did not converge within `gradient_smoothing_max_iter` iterations.                                                                                                                          |
| PIQP_NUMERICS         |  -8   | Numerical error occurred during the factorization or solve of the KKT system.                                                                                                                                 |
| PIQP_UNSOLVED         |  -9   | No solution to differentiate: no `solve` of the current problem data which returned `PIQP_SOLVED` or `PIQP_MAX_ITER_REACHED`, or a provided `forward_result` does not have one of these statuses.             |
| PIQP_INVALID_SETTINGS |  -10  | The solver is not set up, `enable_backward` was not set before `setup`, a gradient w.r.t. the solution has wrong dimensions, or the provided `forward_result` is invalid (dimensions, `rho`/`delta`, `s`/`z`). |
