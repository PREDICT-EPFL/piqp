---
title: CVXPY
layout: default
nav_order: 5
parent: Interfaces
---

Since [CVXPY](https://www.cvxpy.org/) 1.4, PIQP is a supported solver.

To use PIQP as the solver, solve your problem with

```python
import cvxpy as cp

objective = ...
constraints = ...

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.PIQP)
```

For more detailed information and options see the [CVXPY documentation](https://www.cvxpy.org/tutorial/solvers/index.html).