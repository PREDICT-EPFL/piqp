---
title: Gradients
layout: default
parent: API
nav_order: 3
---

The gradients computed by `backward` (see [Differentiation]({{site.baseurl}}/differentiation)), i.e., the gradients of the loss with respect to the problem data.

| Field | Description                                                                                       |
|:------|:--------------------------------------------------------------------------------------------------|
| `P`   | Gradient w.r.t. `P`, symmetric with both triangles (sparse: pattern of the full symmetric `P`)    |
| `c`   | Gradient w.r.t. `c`                                                                               |
| `A`   | Gradient w.r.t. `A` (sparse: pattern of `A`)                                                      |
| `b`   | Gradient w.r.t. `b`                                                                               |
| `G`   | Gradient w.r.t. `G` (sparse: pattern of `G`), zero rows if both bounds of the row are infinite    |
| `h_l` | Gradient w.r.t. `h_l`, zero at infinite bounds                                                    |
| `h_u` | Gradient w.r.t. `h_u`, zero at infinite bounds                                                    |
| `x_l` | Gradient w.r.t. `x_l`, zero at infinite bounds                                                    |
| `x_u` | Gradient w.r.t. `x_u`, zero at infinite bounds                                                    |

{: .warning }
The gradients are only valid if the last `backward` returned `PIQP_BWD_SOLVED` and the problem was not changed by `setup` or `update` since, which can be queried with `gradients_valid`.

## Status code

{% root_include _common/backward_status_code_table.md %}
