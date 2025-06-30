---
title: Getting Started
layout: default
parent: Matlab / Octave
nav_order: 3
---

{% root_include _common/problem_formulation.md %}

## Problem Data

{% root_include _common/problem_data_intro.md %}

To use the Matlab interface of PIQP make sure it is properly added to the path.
We can then define the problem data as

```matlab
P = [6 0; 0 4];
c = [-1; -4];
A = [1 -2];
b = 1;
G = [1 -1; 2, 0];
h_l = [-Inf; -Inf];
h_u = [0.2; -1];
x_l = [-1; -Inf];
x_u = [1; Inf];
```

For the sparse interface $$P$$, $$A$$, and $$G$$ should be sparse matrices.

```matlab
P = sparse([6 0; 0 4]);
A = sparse([1 -2]);
G = sparse([1 -1; 2, 0]);
```

## Solver Instantiation

You can instantiate a solver object using

```matlab
% for dense problems
solver = piqp('dense');
% or for sparse problems
solver = piqp('sparse');
```

## Settings

Settings can be updated using the `update_settings` method:

```matlab
solver.update_settings('verbose', true, 'compute_timings', true);
```

In this example we enable the verbose output and internal timings. The full set of configuration options can be found [here]({{site.baseurl}}/api/settings).

## Solving the Problem

We can now set up the problem using

```matlab
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
```

The data is internally copied, and the solver initializes all internal data structures.

Now, the problem can be solver using

```matlab
result = solver.solve()
```

The result of the optimization are directly returned. More specifically, the most important information includes
* `result.x`: primal solution
* `result.y`: dual solution of equality constraints
* `result.z_l`: dual solution of lower inequality constraints
* `result.z_u`: dual solution of upper inequality constraints
* `result.z_bl`: dual solution of lower bound box constraints
* `result.z_bu`: dual solution of upper bound box constraints
* `result.info.staus`: solver status
* `result.info.primal_obj`: primal objective value
* `result.info.run_time`: total runtime

A detailed list of elements in the results object can be found [here]({{site.baseurl}}/api/result).

{: .warning }
Timing information like `result.info.run_time` is only measured if `compute_timings` is set to `true`.

## Efficient Problem Updates

Instead of creating a new solver object everytime it's possible to update the problem directly using

```matlab
solver.update('P', P_new, 'A', A_new, 'b', b_new, ...);
```

with a subsequent call to

```matlab
result = solver.solve()
```

This allows the solver to internally reuse memory and factorizations speeding up subsequent solves.

{: .warning }
Note the dimension and sparsity pattern of the problem are not allowed to change when calling the `update` function.
