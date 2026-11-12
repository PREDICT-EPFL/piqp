---
title: Derivatives
layout: default
parent: API
nav_order: 3
---

The results of `vjp` and `jvp` (see [Differentiation]({{site.baseurl}}/differentiation)).

## VJP result

The gradients of the loss with respect to the problem data, computed by `vjp`.

| Field    | Description                                                                                                          |
|:---------|:---------------------------------------------------------------------------------------------------------------------|
| `status` | Status of the gradients, see [status code](#status-code)                                                             |
| `P`      | Gradient w.r.t. `P` as a symmetric matrix (dense: both triangles, sparse: pattern of the `P` passed to `setup`)      |
| `c`      | Gradient w.r.t. `c`                                                                                                  |
| `A`      | Gradient w.r.t. `A` (sparse: pattern of the `A` passed to `setup`)                                                   |
| `b`      | Gradient w.r.t. `b`                                                                                                  |
| `G`      | Gradient w.r.t. `G` (sparse: pattern of the `G` passed to `setup`), zero rows if both bounds of the row are infinite |
| `h_l`    | Gradient w.r.t. `h_l`, zero at infinite bounds                                                                       |
| `h_u`    | Gradient w.r.t. `h_u`, zero at infinite bounds                                                                       |
| `x_l`    | Gradient w.r.t. `x_l`, zero at infinite bounds                                                                       |
| `x_u`    | Gradient w.r.t. `x_u`, zero at infinite bounds                                                                       |

## JVP result

The tangents of the solution along the perturbation of the problem data, computed by `jvp`. The fields have the same dimensions as in the [result]({{site.baseurl}}/api/result).

| Field    | Description                                                                 |
|:---------|:----------------------------------------------------------------------------|
| `status` | Status of the tangents, see [status code](#status-code)                     |
| `x`      | Tangent of the primal solution                                              |
| `y`      | Tangent of the dual solution of the equality constraints                    |
| `z_l`    | Tangent of the dual solution of the lower inequality constraints            |
| `z_u`    | Tangent of the dual solution of the upper inequality constraints            |
| `z_bl`   | Tangent of the dual solution of the lower bound box constraints             |
| `z_bu`   | Tangent of the dual solution of the upper bound box constraints             |
| `s_l`    | Tangent of the slack of the lower inequality constraints                    |
| `s_u`    | Tangent of the slack of the upper inequality constraints                    |
| `s_bl`   | Tangent of the slack of the lower bound box constraints                     |
| `s_bu`   | Tangent of the slack of the upper bound box constraints                     |

The tangents of the duals and slacks are zero at infinite bounds and on rows with both bounds infinite.

{: .warning }
The results are only valid if `status` is `PIQP_SOLVED`. `status` is the status returned by the last `vjp` or `jvp`, respectively, and is reset to `PIQP_UNSOLVED` by `setup`, `update`, and `solve`.

## Status code

{% root_include _common/derivative_status_code_table.md %}
