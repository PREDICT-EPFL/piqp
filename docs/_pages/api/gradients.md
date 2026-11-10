---
title: Gradients
layout: default
parent: API
nav_order: 3
---

The gradients computed by `backward` (see [Differentiation]({{site.baseurl}}/differentiation)), i.e., the gradients of the loss with respect to the problem data.

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

{: .warning }
The gradients are only valid if `status` is `PIQP_SOLVED`. `status` is the status returned by the last `backward`, and is reset to `PIQP_UNSOLVED` by `setup` and `update`.

## Status code

{% root_include _common/backward_status_code_table.md %}
