# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import piqp
import numpy as np
from scipy import sparse

P = sparse.csc_matrix([[6, 0], [0, 4]], dtype=np.float64)
c = np.array([-1, -4], dtype=np.float64)
A = sparse.csc_matrix([[1, -2]], dtype=np.float64)
b = np.array([1], dtype=np.float64)
G = sparse.csc_matrix([[1, -1], [2, 0]], dtype=np.float64)
h_u = np.array([0.2, -1], dtype=np.float64)
x_l = np.array([-1, -np.inf], dtype=np.float64)
x_u = np.array([1, np.inf], dtype=np.float64)

solver = piqp.SparseSolver()
solver.settings.verbose = True
solver.settings.compute_timings = True
solver.setup(P, c, A, b, G, None, h_u, x_l, x_u)
status = solver.solve()

print(f'status = {status}')
print(f'x = {solver.result.x}')
