# This file is part of PIQP.
#
# Copyright (c) 2026 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

# The correctness of the gradients is tested in C++, these tests only check the Python interface.

import numpy as np
import pytest
from scipy import sparse
import piqp

# QP with active and inactive inequalities and bounds, and a disabled row (both bounds infinite)
P = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
c = np.array([-1.0, -2.0, 1.0])
A = np.array([[1.0, 1.0, 1.0]])
b = np.array([1.0])
G = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 2.0, 3.0]])
h_l = np.array([-np.inf, -5.0, -np.inf])
h_u = np.array([-1.0, 5.0, np.inf])
x_l = np.array([-np.inf, -np.inf, 0.2])
x_u = np.array([3.0, np.inf, np.inf])

OUTPUTS = ["x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"]
VECTORS = dict(c=c, b=b, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)


def make_solver(dense, vectors=VECTORS, enable_backward=True):
    v = vectors
    solver = piqp.DenseSolver() if dense else piqp.SparseSolver()
    solver.settings.eps_abs = 1e-12
    solver.settings.eps_rel = 1e-12
    solver.settings.enable_backward = enable_backward
    if dense:
        solver.setup(P, v["c"], A, v["b"], G, v["h_l"], v["h_u"], v["x_l"], v["x_u"])
    else:
        solver.setup(sparse.csc_matrix(np.triu(P)), v["c"], sparse.csc_matrix(A), v["b"], sparse.csc_matrix(G),
                     v["h_l"], v["h_u"], v["x_l"], v["x_u"])
    return solver


def solve(dense, vectors):
    solver = make_solver(dense, vectors)
    assert solver.solve() == piqp.PIQP_SOLVED
    return {name: np.array(getattr(solver.result, name)) for name in OUTPUTS}


@pytest.mark.parametrize("dense", [True, False])
def test_backward_arguments(dense):
    # every keyword argument is the gradient w.r.t. the output of the same name, checked with the
    # directional derivative of the loss w^T output along a random direction d of the vector data
    rng = np.random.default_rng(0)
    d = {k: np.where(np.isinf(v), 0.0, rng.standard_normal(v.shape)) for k, v in VECTORS.items()}
    eps = 1e-6
    plus = solve(dense, {k: v + eps * d[k] for k, v in VECTORS.items()})
    minus = solve(dense, {k: v - eps * d[k] for k, v in VECTORS.items()})

    solver = make_solver(dense)
    assert solver.solve() == piqp.PIQP_SOLVED
    for name in OUTPUTS:
        w = rng.standard_normal(plus[name].shape)
        w[np.isinf(plus[name])] = 0.0  # infinite slacks
        assert solver.backward(**{"grad_" + name: w}) == piqp.PIQP_SOLVED
        derivative = sum(np.asarray(getattr(solver.gradients, k)) @ d[k] for k in VECTORS)
        fd = w @ (plus[name] - minus[name]) / (2 * eps)
        assert derivative == pytest.approx(fd, abs=1e-6, rel=1e-4), name


@pytest.mark.parametrize("dense", [True, False])
def test_backward_forward_result(dense):
    solver = make_solver(dense)
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=np.ones(3)) == piqp.PIQP_SOLVED
    expected = np.array(solver.gradients.c)

    # solver.result refers to the memory of the solver
    result = piqp.Result()
    for name in OUTPUTS:
        setattr(result, name, np.array(getattr(solver.result, name)))
    result.info = solver.result.info
    assert solver.backward(grad_x=np.ones(3), forward_result=result) == piqp.PIQP_SOLVED
    np.testing.assert_allclose(solver.gradients.c, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dense", [True, False])
def test_backward_status(dense):
    solver = make_solver(dense, enable_backward=False)
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward() == piqp.PIQP_INVALID_SETTINGS

    solver = make_solver(dense)
    grads = solver.gradients
    assert grads.status == piqp.PIQP_UNSOLVED
    assert solver.backward() == piqp.PIQP_UNSOLVED
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward() == piqp.PIQP_SOLVED
    assert grads.status == piqp.PIQP_SOLVED

    assert solver.backward(grad_x=np.zeros(4)) == piqp.PIQP_INVALID_SETTINGS
    assert grads.status == piqp.PIQP_INVALID_SETTINGS

    result = piqp.Result()
    result.info.status = piqp.PIQP_PRIMAL_INFEASIBLE
    assert solver.backward(forward_result=result) == piqp.PIQP_UNSOLVED
    assert grads.status == piqp.PIQP_UNSOLVED

    # the smoothing settings are passed on
    solver.settings.gradient_smoothing = True
    solver.settings.gradient_smoothing_max_iter = 1
    assert solver.backward() == piqp.PIQP_MAX_ITER_REACHED
    solver.settings.gradient_smoothing = False

    assert solver.backward() == piqp.PIQP_SOLVED
    solver.update(c=2.0 * c)
    assert grads.status == piqp.PIQP_UNSOLVED


@pytest.mark.parametrize("dense", [True, False])
def test_gradient_types(dense):
    solver = make_solver(dense)
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=np.ones(3)) == piqp.PIQP_SOLVED
    grads = solver.gradients
    for name, shape in [("P", (3, 3)), ("A", (1, 3)), ("G", (3, 3))]:
        mat = getattr(grads, name)
        assert mat.shape == shape
        if dense:
            assert isinstance(mat, np.ndarray)
            assert not mat.flags.writeable
        else:
            assert isinstance(mat, sparse.csc_matrix)
    for name, size in [("c", 3), ("b", 1), ("h_l", 3), ("h_u", 3), ("x_l", 3), ("x_u", 3)]:
        vec = getattr(grads, name)
        assert vec.shape == (size,)
        assert not vec.flags.writeable


def test_sparse_gradients_are_views():
    solver = make_solver(False)
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=np.ones(3)) == piqp.PIQP_SOLVED
    gP, gG = solver.gradients.P, solver.gradients.G
    assert not gP.data.flags.writeable
    with pytest.raises(ValueError):
        gP.data[0] = 1.0
    # the sparsity pattern of the P passed to setup(), here upper triangular
    P_triu = sparse.csc_matrix(np.triu(P))
    np.testing.assert_array_equal(gP.indptr, P_triu.indptr)
    np.testing.assert_array_equal(gP.indices, P_triu.indices)
    # the usual operations work on the read-only views
    np.testing.assert_allclose(gG @ np.ones(3), gG.toarray() @ np.ones(3))

    # the views reflect the next backward(), which is linear in the arguments
    gP_before = gP.toarray()
    assert solver.backward(grad_x=-np.ones(3)) == piqp.PIQP_SOLVED
    np.testing.assert_allclose(gP.toarray(), -gP_before, atol=1e-12)

    # setup() with the same sparsity pattern keeps the storage, the views stay valid
    solver.setup(sparse.csc_matrix(np.triu(P)), 2.0 * c, sparse.csc_matrix(A), b, sparse.csc_matrix(G), h_l, h_u, x_l, x_u)
    assert np.shares_memory(gP.data, solver.gradients.P.data)


if __name__ == "__main__":
    pytest.main([__file__])
