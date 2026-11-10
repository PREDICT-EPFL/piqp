# This file is part of PIQP.
#
# Copyright (c) 2026 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from scipy import sparse
import piqp

# Strictly complementary QP satisfying LICQ: G row 0 is active at its upper bound,
# G row 1 is inactive, G row 2 is disabled (both bounds infinite), x_2 is active
# at its lower bound, and x_0 has an inactive upper bound. Finite differences
# therefore do not cross an active-set change.
P = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
c = np.array([-1.0, -2.0, 1.0])
A = np.array([[1.0, 1.0, 1.0]])
b = np.array([1.0])
G = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 2.0, 3.0]])
h_l = np.array([-np.inf, -5.0, -np.inf])
h_u = np.array([-1.0, 5.0, np.inf])
x_l = np.array([-np.inf, -np.inf, 0.2])
x_u = np.array([3.0, np.inf, np.inf])

# loss L = w_x^T x + w_y^T y + w_z^T z_u
w_x = np.array([1.0, -2.0, 0.5])
w_y = np.array([0.3])
w_z = np.array([0.7, -0.4, 0.0])

EPS = 1e-6


def make_solver(dense, data):
    solver = piqp.DenseSolver() if dense else piqp.SparseSolver()
    solver.settings.eps_abs = 1e-12
    solver.settings.eps_rel = 1e-12
    solver.settings.enable_backward = True
    if dense:
        solver.setup(data["P"], data["c"], data["A"], data["b"], data["G"],
                     data["h_l"], data["h_u"], data["x_l"], data["x_u"])
    else:
        solver.setup(sparse.csc_matrix(np.triu(data["P"])), data["c"], sparse.csc_matrix(data["A"]),
                     data["b"], sparse.csc_matrix(data["G"]), data["h_l"], data["h_u"], data["x_l"], data["x_u"])
    return solver


def base_data():
    return dict(P=P.copy(), c=c.copy(), A=A.copy(), b=b.copy(), G=G.copy(),
                h_l=h_l.copy(), h_u=h_u.copy(), x_l=x_l.copy(), x_u=x_u.copy())


def loss(dense, data):
    solver = make_solver(dense, data)
    assert solver.solve() == piqp.PIQP_SOLVED
    res = solver.result
    return w_x @ res.x + w_y @ res.y + w_z @ res.z_u


def finite_difference(dense, name, idx, symmetric=False):
    data_p = base_data()
    data_m = base_data()
    data_p[name][idx] += EPS
    data_m[name][idx] -= EPS
    if symmetric and idx[0] != idx[1]:
        data_p[name][idx[::-1]] += EPS
        data_m[name][idx[::-1]] -= EPS
    return (loss(dense, data_p) - loss(dense, data_m)) / (2 * EPS)


def to_dense(mat):
    return mat.toarray() if sparse.issparse(mat) else np.asarray(mat)


@pytest.mark.parametrize("dense", [True, False])
def test_backward_finite_differences(dense):
    solver = make_solver(dense, base_data())
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x, grad_y=w_y, grad_z_u=w_z) == piqp.PIQP_SOLVED
    assert solver.gradients.status == piqp.PIQP_SOLVED
    grads = solver.gradients

    for name in ["c", "b", "h_l", "h_u", "x_l", "x_u"]:
        bound = base_data()[name]
        grad = np.asarray(getattr(grads, name))
        for i in range(bound.shape[0]):
            if np.isinf(bound[i]):
                assert grad[i] == 0.0
            else:
                assert grad[i] == pytest.approx(finite_difference(dense, name, (i,)), abs=1e-5, rel=1e-4)

    for name in ["P", "A", "G"]:
        mat = base_data()[name]
        grad = to_dense(getattr(grads, name))
        assert grad.shape == mat.shape
        for i in range(mat.shape[0]):
            # G row 2 is disabled, its gradient is zero
            if name == "G" and i == 2:
                assert np.all(grad[i] == 0.0)
                continue
            for j in range(mat.shape[1]):
                if not dense and mat[i, j] == 0.0:
                    continue  # sparse gradients only live on the sparsity pattern
                if name == "P":
                    if j < i:
                        continue
                    expected = finite_difference(dense, name, (i, j), symmetric=True)
                    # the gradient w.r.t. the symmetric P is 0.5 * (lam_x x^T + x lam_x^T), the symmetric perturbation
                    # of the entries (i, j) and (j, i) counts it twice (the sparse gradient is upper triangular here)
                    analytic = 2.0 * grad[i, j] if i != j else grad[i, j]
                else:
                    expected = finite_difference(dense, name, (i, j))
                    analytic = grad[i, j]
                assert analytic == pytest.approx(expected, abs=1e-5, rel=1e-4), (name, i, j)


@pytest.mark.parametrize("dense", [True, False])
def test_backward_forward_result(dense):
    solver = make_solver(dense, base_data())
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_SOLVED
    expected = np.array(solver.gradients.c)

    result = piqp.Result()
    for field in ["x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"]:
        setattr(result, field, np.array(getattr(solver.result, field)))
    result.info = solver.result.info

    # the default call reuses the factorization of the forward solve (second-to-last iterate),
    # the forward result is refactorized, so both agree up to the solver accuracy
    assert solver.backward(grad_x=w_x, forward_result=result) == piqp.PIQP_SOLVED
    np.testing.assert_allclose(solver.gradients.c, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dense", [True, False])
def test_backward_smoothing(dense):
    solver = make_solver(dense, base_data())
    solver.settings.gradient_smoothing = True
    solver.settings.gradient_smoothing_mu = 1e-3
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_SOLVED
    assert np.all(np.isfinite(solver.gradients.c))


@pytest.mark.parametrize("dense", [True, False])
def test_backward_requires_enable_backward(dense):
    solver = make_solver(dense, base_data())
    solver.settings.enable_backward = False
    data = base_data()
    if dense:
        solver.setup(data["P"], data["c"])
    else:
        solver.setup(sparse.csc_matrix(np.triu(data["P"])), data["c"])
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_INVALID_SETTINGS
    assert solver.gradients.status == piqp.PIQP_INVALID_SETTINGS


@pytest.mark.parametrize("dense", [True, False])
def test_backward_failures(dense):
    solver = make_solver(dense, base_data())
    assert solver.gradients.status == piqp.PIQP_UNSOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_UNSOLVED
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_SOLVED
    grads = solver.gradients
    assert grads.status == piqp.PIQP_SOLVED

    # a failed call invalidates the previously computed gradients, also in previously obtained references
    assert solver.backward(grad_x=np.zeros(4)) == piqp.PIQP_INVALID_SETTINGS
    assert grads.status == piqp.PIQP_INVALID_SETTINGS

    # a forward result which is not solved
    result = piqp.Result()
    result.info.status = piqp.PIQP_PRIMAL_INFEASIBLE
    assert solver.backward(grad_x=w_x, forward_result=result) == piqp.PIQP_UNSOLVED
    assert solver.gradients.status == piqp.PIQP_UNSOLVED

    # a malformed forward result
    result.info = solver.result.info
    assert solver.backward(grad_x=w_x, forward_result=result) == piqp.PIQP_INVALID_SETTINGS
    assert solver.gradients.status == piqp.PIQP_INVALID_SETTINGS

    assert solver.backward(grad_x=w_x) == piqp.PIQP_SOLVED
    solver.update(c=c)
    assert solver.gradients.status == piqp.PIQP_UNSOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_UNSOLVED



def test_sparse_gradients_are_views():
    data = base_data()
    solver = make_solver(False, data)
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_SOLVED
    grads = solver.gradients
    gP, gA, gG = grads.P, grads.A, grads.G
    for M in (gP, gA, gG):
        assert isinstance(M, sparse.csc_matrix)
        assert not M.data.flags.writeable
    # the sparsity pattern of the P passed to setup(), here upper triangular
    P_triu = sparse.csc_matrix(np.triu(data["P"]))
    np.testing.assert_array_equal(gP.indptr, P_triu.indptr)
    np.testing.assert_array_equal(gP.indices, P_triu.indices)
    # read-only, but the usual operations work on the views
    with pytest.raises(ValueError):
        gP.data[0] = 1.0
    np.testing.assert_allclose(gP.sum(), gP.toarray().sum())
    np.testing.assert_allclose(gG @ np.ones(3), gG.toarray() @ np.ones(3))

    # the held matrices reflect the next backward(), which is linear in the gradients w.r.t. the solution
    gP_before, gG_before = gP.toarray(), gG.toarray()
    assert solver.backward(grad_x=-w_x) == piqp.PIQP_SOLVED
    np.testing.assert_allclose(gP.toarray(), -gP_before, atol=1e-12)
    np.testing.assert_allclose(gG.toarray(), -gG_before, atol=1e-12)

    # setup() with the same sparsity pattern keeps the storage, the views stay valid
    data["c"] = 2.0 * c
    solver.setup(sparse.csc_matrix(np.triu(data["P"])), data["c"], sparse.csc_matrix(data["A"]),
                 data["b"], sparse.csc_matrix(data["G"]), data["h_l"], data["h_u"], data["x_l"], data["x_u"])
    assert np.shares_memory(gP.data, solver.gradients.P.data)
    assert solver.solve() == piqp.PIQP_SOLVED
    assert solver.backward(grad_x=w_x) == piqp.PIQP_SOLVED
    np.testing.assert_array_equal(gP.toarray(), solver.gradients.P.toarray())


if __name__ == "__main__":
    pytest.main([__file__])
