---
title: Getting Started with C++
layout: default
parent: C/C++
nav_order: 2
---

{% root_include _common/problem_formulation.md %}

## Problem Data

{% root_include _common/problem_data_intro.md %}

The C++ interface uses Eigen and is automatically included in the PIQP header files
```c++
#include <limits>
#include "piqp/piqp.hpp"
```
`<limits>` is included for getting infinity values.

We can then define the problem data as

```c++
int n = 2;
int p = 1;
int m = 2;

double inf = std::numeric_limits<double>::infinity();

Eigen::MatrixXd P(n, n); P << 6, 0, 0, 4;
Eigen::VectorXd c(n); c << -1, -4;

Eigen::MatrixXd A(p, n); A << 1, -2;
Eigen::VectorXd b(p); b << 1;

Eigen::MatrixXd G(m, n); G << 1, -1, 2, 0;
Eigen::VectorXd h_l(m); h << -inf, -inf;
Eigen::VectorXd h_u(m); h << 0.2, -1;

Eigen::VectorXd x_l(n); x_lb << -1, -inf;
Eigen::VectorXd x_u(n); x_ub << 1, inf;
```

For the sparse interface $$P$$, $$A$$, and $$G$$ have to be in compressed sparse column (CSC) format.

```c++
Eigen::MatrixXd P_dense(n, n); P_dense << 6, 0, 0, 4;
Eigen::SparseMatrix<double> P = P_dense.sparseView();

Eigen::MatrixXd A_dense(p, n); A_dense << 1, -2;
Eigen::SparseMatrix<double> A = A_dense.sparseView();

Eigen::MatrixXd G_dense(m, n); G_dense << 1, -1, 2, 0;
Eigen::SparseMatrix<double> G = G_dense.sparseView();
```

{: .note }
> There are better options to build sparse matrices in Eigen. For example, we can also build the sparse matrix $$P$$ using the `insert` API:
```c++
Eigen::SparseMatrix<double> P(n, n);
P.insert(0, 0) = 6;
P.insert(1, 1) = 4;
P.makeCompressed();
```
> For more details see the [Eigen Docs](https://eigen.tuxfamily.org/dox/group__TutorialSparse.html).

## Solver Instantiation

You can instantiate a solver object using

```c++
// for dense problems
piqp::DenseSolver<double> solver;
// or for sparse problems
piqp::SparseSolver<double> solver;
```

where the template argument defines the data type, i.e., `double` or `float`.

## Settings

Settings can be directly set on the solver object:

```c++
solver.settings().verbose = true;
solver.settings().compute_timings = true;
```

In this example we enable the verbose output and internal timings. The full set of configuration options can be found [here]({{site.baseurl}}/api/settings).

## Solving the Problem

We can now set up the problem using

```c++
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
```

{: .note }
Every variable except `P` and `c` are optional and `piqp::nullopt` may be passed.

The data is internally copied, and the solver initializes all internal data structures.

Now, the problem can be solver using

```c++
piqp::Status status = solver.solve();
```

### Status code

{% root_include _common/status_code_table.md %}

## Extracting the Result

The result of the optimization can be obtained from the `solver.result()` object. More specifically, the most important information includes
* `solver.result().x`: primal solution
* `solver.result().y`: dual solution of equality constraints
* `solver.result().z_l`: dual solution of lower inequality constraints
* `solver.result().z_u`: dual solution of upper inequality constraints
* `solver.result().z_bl`: dual solution of lower bound box constraints
* `solver.result().z_bu`: dual solution of upper bound box constraints
* `solver.result().info.primal_obj`: primal objective value
* `solver.result().info.run_time`: total runtime 

A detailed list of elements in the results object can be found [here]({{site.baseurl}}/api/result).

{: .warning }
Timing information like `solver.result().info.run_time` is only measured if `solver.settings().compute_timings` is set to `true`.

## Efficient Problem Updates

Instead of creating a new solver object everytime it's possible to update the problem directly using

```c++
solver.update(P, c, A, b, G, h_l, h_u, x_l, x_u);
```

with a subsequent call to 

```c++
piqp::Status status = solver.solve();
```

This allows the solver to internally reuse memory and factorizations speeding up subsequent solves. Similar to the `setup` function, all parameters are optional and `piqp::nullopt` may be passed instead.

{: .warning }
Note the dimension and sparsity pattern of the problem are not allowed to change when calling the `update` function.
