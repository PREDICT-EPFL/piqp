---
title: Getting Started
layout: default
parent: Python
nav_order: 2
---

{% root_include _common/problem_formulation.md %}

## Problem Data

{% root_include _common/problem_data_intro.md %}

To use the Python interface of PIQP import the following packages:
```python
import piqp
import numpy as np
from scipy import sparse
```
`scipy` is only needed if the sparse interface is used.

We can then define the problem data as

```python
P = np.array([[6, 0], [0, 4]], dtype=np.float64)
c = np.array([-1, -4], dtype=np.float64)
A = np.array([[1, -2]], dtype=np.float64)
b = np.array([1], dtype=np.float64)
G = np.array([[1, -1], [2, 0]], dtype=np.float64)
h_l = np.array([-np.inf, -np.inf], dtype=np.float64)
h_u = np.array([0.2, -1], dtype=np.float64)
x_l = np.array([-1, -np.inf], dtype=np.float64)
x_u = np.array([1, np.inf], dtype=np.float64)
```

For the sparse interface $$P$$, $$A$$, and $$G$$ have to be in compressed sparse column (CSC) format.

```python
P = sparse.csc_matrix([[6, 0], [0, 4]], dtype=np.float64)
A = sparse.csc_matrix([[1, -2]], dtype=np.float64)
G = sparse.csc_matrix([[1, -1], [2, 0]], dtype=np.float64)
```

## Solver Instantiation

You can instantiate a solver object using

```python
// for dense problems
solver = piqp.DenseSolver()
// or for sparse problems
solver = piqp.SparseSolver()
```

## Settings

Settings can be directly set on the solver object:

```python
solver.settings.verbose = True
solver.settings.compute_timings = True
```

In this example we enable the verbose output and internal timings. The full set of configuration options can be found [here]({{site.baseurl}}/api/settings).

## Solving the Problem

We can now set up the problem using

```python
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u)
```

{: .note }
Every variable except `P` and `c` are optional and `None` may be passed.

The data is internally copied, and the solver initializes all internal data structures.

Now, the problem can be solver using

```python
status = solver.solve()
```

### Status code

{% root_include _common/status_code_table.md %}

## Extracting the Result

The result of the optimization can be obtained from the `solver.result` object. More specifically, the most important information includes
* `solver.result.x`: primal solution
* `solver.result.y`: dual solution of equality constraints
* `solver.result.z_l`: dual solution of lower inequality constraints
* `solver.result.z_u`: dual solution of upper inequality constraints
* `solver.result.z_bl`: dual solution of lower bound box constraints
* `solver.result.z_bu`: dual solution of upper bound box constraints
* `solver.result.info.primal_obj`: primal objective value
* `solver.result.info.run_time`: total runtime

A detailed list of elements in the results object can be found [here]({{site.baseurl}}/api/result).

{: .warning }
Timing information like `solver.result.info.run_time` is only measured if `solver.settings.compute_timings` is set to `true`.

## Efficient Problem Updates

Instead of creating a new solver object everytime it's possible to update the problem directly using

```python
solver.update(P, c, A, b, G, h_l, h_u, x_l, x_u)
```

with a subsequent call to

```python
status = solver.solve()
```

This allows the solver to internally reuse memory and factorizations speeding up subsequent solves. Similar to the `setup` function, all parameters are optional and `None` may be passed instead.

{: .warning }
Note the dimension and sparsity pattern of the problem are not allowed to change when calling the `update` function.
