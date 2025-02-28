# Maros Meszaros problems data files in .mat format

These are the converted Maros Meszaros problems to .mat files.
The problems have the form
$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top P x + c^\top x \\
\text {s.t.}\quad & Ax=b, \\
& Gx \leq h, \\
& x_{lb} \leq x \leq x_{ub},
\end{aligned}
$$

These files are converted from [osqp_benchmark](https://github.com/osqp/osqp_benchmarks/tree/master/problem_classes/maros_meszaros_data) (see LICENSE file). To generate the files in this folder copy the `.mat` files from [osqp_benchmark](https://github.com/osqp/osqp_benchmarks/tree/master/problem_classes/maros_meszaros_data) into a folder `osqp_maros_meszaros_data` outside this folder and run the `convert_osqp_to_piqp.m` script.