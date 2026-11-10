| Status Code                     | Value | Description                                                                                                     |
|:--------------------------------|:-----:|:----------------------------------------------------------------------------------------------------------------|
| PIQP_BWD_SOLVED                 |   1   | Gradients computed successfully.                                                                                |
| PIQP_BWD_INVALID_SETTINGS       |  -1   | `enable_backward` was not set before `setup`, or a gradient w.r.t. the solution has wrong dimensions.           |
| PIQP_BWD_INVALID_FORWARD_RESULT |  -2   | No valid forward solution: no `solve` of the current problem data, or an invalid `forward_result` was provided. |
| PIQP_BWD_NUMERICS               |  -3   | Numerical error occurred during the factorization or solve of the KKT system.                                   |
| PIQP_BWD_MAX_ITER_REACHED       |  -4   | Gradient smoothing did not converge within `gradient_smoothing_max_iter` iterations.                            |
