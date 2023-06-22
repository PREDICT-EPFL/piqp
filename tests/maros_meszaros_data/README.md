# Maros Meszaros problems data files in .mat format

These are the converted Maros Meszaros problems to .mat files.
The problems have the form
```
minimize        0.5 x' P x + q' x + r

subject to      l <= A x <= u
```

where `x in R^n` is the optimization variable. The objective function is defined by a positive semidefinite matrix `P in S^n_+`, a vector `q in R^n` and a scalar `r in R`. The linear constraints are defined by matrix `A in R^{m x n}` and vectors `l in R^m U {-inf}^m`, `u in R^m U {+inf}^m`.

These files are taken from [osqp_benchmark](https://github.com/osqp/osqp_benchmarks/tree/master/problem_classes/maros_meszaros_data) (see LICENSE file).