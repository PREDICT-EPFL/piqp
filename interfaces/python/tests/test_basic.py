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
    G = sparse.csc_matrix([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    h = np.array([0.7, 0.7, 0, 0], dtype=np.float64)

    start_time = time.time()

    solver = piqp.SparseSolver()
    solver.settings.verbose = True
    solver.settings.compute_timings = True
    solver.setup(P, c, A, b, G, h)
    solver.update(None, c, A, None, G, h)
    solver.solve()

    end_time = time.time()
    print(f'Execution time python sparse: {end_time - start_time:.3e}s')

    P = P.todense()
    c = np.array([1, 1], dtype=np.float64)
    A = A.todense()
    b = np.array([1], dtype=np.float64)
    G = G.todense()
    h = np.array([0.7, 0.7, 0, 0], dtype=np.float64)

    start_time = time.time()

    solver = piqp.DenseSolver()
    solver.settings.verbose = True
    solver.settings.compute_timings = True
    solver.setup(P, c, A, b, G, h)
    solver.update(None, c, A, None, G, h)
    solver.solve()

    end_time = time.time()
    print(f'Execution time python dense: {end_time - start_time:.3e}s')


if __name__ == '__main__':
    test_main()
