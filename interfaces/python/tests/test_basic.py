# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from scipy import sparse
import piqp


def test_main():
    P = sparse.csc_matrix([[4, 1], [1, 2]], dtype=np.float64)
    c = np.array([1, 1], dtype=np.float64)
    A = sparse.csc_matrix([[1, 1]], dtype=np.float64)
    b = np.array([1], dtype=np.float64)
    G = sparse.csc_matrix([[1, 0], [-1, 0]], dtype=np.float64)
    h_u = np.array([0.7, 0], dtype=np.float64)
    x_l = np.array([-np.inf, 0], dtype=np.float64)
    x_u = np.array([np.inf, 0.7], dtype=np.float64)

    start_time = time.time()

    solver = piqp.SparseSolver()
    solver.settings.verbose = True
    solver.settings.compute_timings = True
    solver.setup(P, c, A, b, G, None, h_u, x_l, x_u)
    solver.update(None, c, A, None, G, None, h_u, None, x_u)
    solver.solve()

    end_time = time.time()
    print(f'Execution time python sparse: {end_time - start_time:.3e}s')

    P = P.todense()
    A = A.todense()
    G = G.todense()

    start_time = time.time()

    solver = piqp.DenseSolver()
    solver.settings.verbose = True
    solver.settings.compute_timings = True
    solver.setup(P, c, A, b, G, None, h_u, x_l, x_u)
    solver.update(None, c, A, None, G, None, h_u, None, x_u)
    solver.solve()

    end_time = time.time()
    print(f'Execution time python dense: {end_time - start_time:.3e}s')


if __name__ == '__main__':
    test_main()
