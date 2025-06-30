---
title: Getting Started with C
layout: default
parent: C/C++
nav_order: 3
---

{% root_include _common/problem_formulation.md %}

## Problem Data

{% root_include _common/problem_data_intro.md %}

To include the C interface of PIQP include the following headers
```c
#include "stdlib.h"
#include "piqp.h"
```
`stdlib.h` is later needed to allocate necessary structs.

We can then define the problem data as

```c
piqp_int n = 2;
piqp_int p = 1;
piqp_int m = 2;

piqp_float P[4] = {6, 0, 0, 4};
piqp_float c[2] = {-1, -4};

piqp_float A[2] = {1, -2};
piqp_float b[1] = {1};

piqp_float G[4] = {1, -1, 2, 0};
piqp_float h_l[2] = {-PIQP_INF, -PIQP_INF};
piqp_float h_u[2] = {0.2, -1};

piqp_float x_l[2] = {-1, -PIQP_INF};
piqp_float x_u[2] = {1, PIQP_INF};

piqp_data_sparse* data = (piqp_data_sparse*) malloc(sizeof(piqp_data_sparse));
data->n = n;
data->p = p;
data->m = m;
data->P = P;
data->c = c;
data->A = A;
data->b = b;
data->G = G;
data->h_l = h_l;
data->h_u = h_u;
data->x_l = x_l;
data->x_u = x_u;
```

Here `PIQP_INF` represents $$\infty$$, and we store the whole problem in the `data` struct.

For the sparse interface $$P$$, $$A$$, and $$G$$ have to be in compressed sparse column (CSC) format.

```c
piqp_float P_x[2] = {6, 4};
piqp_int P_nnz = 2;
piqp_int P_p[3] = {0, 1, 2};
piqp_int P_i[2] = {0, 1};

piqp_float A_x[2] = {1, -2};
piqp_int A_nnz = 2;
piqp_int A_p[3] = {0, 1, 2};
piqp_int A_i[2] = {0, 0};

piqp_float G_x[3] = {1, 2, -1};
piqp_int G_nnz = 3;
piqp_int G_p[3] = {0, 2, 3};
piqp_int G_i[4] = {0, 1, 0};

data->P = piqp_csc_matrix(data->n, data->n, P_nnz, P_p, P_i, P_x);
data->A = piqp_csc_matrix(data->p, data->n, A_nnz, A_p, A_i, A_x);
data->G = piqp_csc_matrix(data->m, data->n, G_nnz, G_p, G_i, G_x);
```

`piqp_csc_matrix(...)` is a helper function allocating a `piqp_csc` struct and filling its fields accordingly.

{: .note }
Every member in `data`, except `P` and `c`, is optional and may be `NULL`.

## Settings

To set custom settings, a `piqp_settings` struct has to be instantiated and the default settings have to be set:

```c
piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));

// dense interface
piqp_set_default_settings_dense(settings);
// sparse interface
piqp_set_default_settings_sparse(settings);
settings->verbose = 1;
settings->compute_timings = 1;
```

In this example we enable the verbose output and internal timings. The full set of configuration options can be found [here]({{site.baseurl}}/api/settings).

## Solving the Problem

We can now set up the problem using

```c
// workspace
piqp_workspace* work;
// dense interface
piqp_setup_dense(&work, data, settings);
// or sparse interface
piqp_setup_sparse(&work, data, settings);
```

The data is internally copied, and the solver initializes all internal data structures. Note that the settings field is optional and `NULL` can be passed.

Now, the problem can be solver using

```c
piqp_status status = piqp_solve(work);
```

### Status code

{% root_include _common/status_code_table.md %}

## Extracting the Result

The result of the optimization can be obtained from the `work->result` struct. More specifically, the most important information includes
* `work->result->x`: primal solution
* `work->result->y`: dual solution of equality constraints
* `work->result->z_l`: dual solution of lower inequality constraints
* `work->result->z_u`: dual solution of upper inequality constraints
* `work->result->z_bl`: dual solution of lower bound box constraints
* `work->result->z_bu`: dual solution of upper bound box constraints
* `work->result->info.primal_obj`: primal objective value
* `work->result->info.run_time`: total runtime

A detailed list of elements in the results object can be found [here]({{site.baseurl}}/api/result).

{: .warning }
Timing information like `work->result->info.run_time` is only measured if `settings->compute_timings` is set to `1`.

## Efficient Problem Updates

Instead of creating a new solver object everytime it's possible to update the problem directly using

```c
// dense interface
piqp_update_dense(&work, P, c, A, b, G, h_l, h_u, x_l, x_u);
// or sparse interface
piqp_update_sparse(&work, P, c, A, b, G, h_l, h_u, x_l, x_u);
```

with a subsequent call to

```c
piqp_status status = piqp_solve(work);
```

This allows the solver to internally reuse memory and factorizations speeding up subsequent solves. Similar to the `piqp_setup_*` functions, all parameters are optional and `NULL` may be passed instead.

{: .warning }
Note the dimension and sparsity pattern of the problem are not allowed to change when calling the `piqp_update_*` functions.
