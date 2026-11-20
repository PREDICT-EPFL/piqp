// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <cstdlib>

#include "piqp.h"
#include "piqp/piqp.hpp"

#include "gtest/gtest.h"

/*
 * first QP:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
 *
 * second QP:
 * min 4 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 2, x1 = 3 x2
*/
TEST(CInterfaceTest, SimpleDenseQPWithUpdate)
{
    piqp_int n = 2;
    piqp_int p = 1;
    piqp_int m = 3;

    piqp_float P[4] = {6, 0, 0, 4};
    piqp_float c[2] = {-1, -4};

    piqp_float A[2] = {1, -2};
    piqp_float b[1] = {0};

    piqp_float G[6] = {1, 0, 1, 0, 1, 0};
    piqp_float h_l[3] = {-1, -PIQP_INF, -2};
    piqp_float h_u[3] = {PIQP_INF, 1, 2};

    piqp_float x_l[2] = {-PIQP_INF, -1};
    piqp_float x_u[2] = {PIQP_INF, 1};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_dense* data = (piqp_data_dense*) malloc(sizeof(piqp_data_dense));

    piqp_set_default_settings_dense(settings);
    settings->verbose = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = P;
    data->c = c;
    data->A = A;
    data->b = b;
    data->G = G;
    data->h_l = h_l;
    data->h_u = h_u;
    data->x_l = x_l;
    data->x_u = x_u;

    piqp_setup_dense(&work, data, settings);
    piqp_status status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.4285714, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.2142857, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.5714286, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    P[0] = 8;
    A[1] = -3;
    h_u[0] = 2;
    x_u[1] = 2;

    piqp_update_dense(work, data->P, NULL, data->A, NULL, NULL, NULL, data->h_u, NULL, data->x_u);
    status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.2763157, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.0921056, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.2105263, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    piqp_cleanup(work);
    if (settings) free(settings);
    if (data) free(data);
}

/*
 * first QP:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
 *
 * second QP:
 * min 4 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 2, x1 = 3 x2
*/
TEST(CInterfaceTest, SimpleSparseQPWithUpdate)
{
    piqp_int n = 2;
    piqp_int p = 1;
    piqp_int m = 3;

    piqp_float P_x[2] = {6, 4};
    piqp_int P_nnz = 2;
    piqp_int P_p[3] = {0, 1, 2};
    piqp_int P_i[2] = {0, 1};
    piqp_float c[2] = {-1, -4};

    piqp_float A_x[2] = {1, -2};
    piqp_int A_nnz = 2;
    piqp_int A_p[3] = {0, 1, 2};
    piqp_int A_i[2] = {0, 0};
    piqp_float b[1] = {0};

    piqp_float G_x[3] = {1, 1, 1};
    piqp_int G_nnz = 3;
    piqp_int G_p[3] = {0, 3, 3};
    piqp_int G_i[3] = {0, 1, 2};
    piqp_float h_l[3] = {-1, -PIQP_INF, -2};
    piqp_float h_u[3] = {PIQP_INF, 1, 2};

    piqp_float x_l[2] = {-PIQP_INF, -1};
    piqp_float x_u[2] = {PIQP_INF, 1};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_sparse* data = (piqp_data_sparse*) malloc(sizeof(piqp_data_sparse));

    piqp_set_default_settings_sparse(settings);
    settings->verbose = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = piqp_csc_matrix(data->n, data->n, P_nnz, P_p, P_i, P_x);
    data->c = c;
    data->A = piqp_csc_matrix(data->p, data->n, A_nnz, A_p, A_i, A_x);
    data->b = b;
    data->G = piqp_csc_matrix(data->m, data->n, G_nnz, G_p, G_i, G_x);
    data->h_l = h_l;
    data->h_u = h_u;
    data->x_l = x_l;
    data->x_u = x_u;

    piqp_setup_sparse(&work, data, settings);
    piqp_status status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.4285714, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.2142857, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.5714286, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    P_x[0] = 8;
    A_x[1] = -3;
    h_u[0] = 2;
    x_u[1] = 2;

    piqp_update_sparse(work, data->P, NULL, data->A, NULL, NULL, NULL, data->h_u, NULL, data->x_u);
    status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.2763157, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.0921056, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.2105263, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    piqp_cleanup(work);
    if (settings) free(settings);
    if (data)
    {
        if (data->P) free(data->P);
        if (data->A) free(data->A);
        if (data->G) free(data->G);
        free(data);
    }
}

/*
 * The vjp() and jvp() tests only check the C interface, i.e., the derivatives have to be exactly the ones of the
 * C++ solver for the same arguments. QP with an active and an inactive inequality, such that the gradient
 * w.r.t. G is not symmetric and its layout is tested:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. x1 = 2 x2, x1 + x2 <= 0.3 (active), x1 >= -1, x2 <= 0.5
 */
namespace vjp_qp
{
constexpr piqp_int n = 2;
constexpr piqp_int p = 1;
constexpr piqp_int m = 2;
piqp_float P[4] = {6, 0, 0, 4}; // row-major
piqp_float c[2] = {-1, -4};
piqp_float A[2] = {1, -2};
piqp_float b[1] = {0};
piqp_float G[4] = {1, 1, 1, 0};
piqp_float h_l[2] = {-PIQP_INF, -1};
piqp_float h_u[2] = {0.3, PIQP_INF};
piqp_float x_l[2] = {-PIQP_INF, -PIQP_INF};
piqp_float x_u[2] = {PIQP_INF, 0.5};

const piqp_float grad_x[2] = {1, 2};
const piqp_float grad_y[1] = {0.5};
const piqp_float grad_z_l[2] = {0.3, -0.2};
const piqp_float grad_z_u[2] = {0.7, 0.1};
const piqp_float grad_z_bl[2] = {-0.4, 0.6};
const piqp_float grad_z_bu[2] = {0.2, -0.3};
const piqp_float grad_s_l[2] = {0.1, 0.4};
const piqp_float grad_s_u[2] = {-0.5, 0.2};
const piqp_float grad_s_bl[2] = {0.3, 0.3};
const piqp_float grad_s_bu[2] = {-0.1, 0.8};

// row-major, the lower triangular part of dP is not read, i.e., a transposed layout would change the result
const piqp_float dP[4] = {0.5, 0.2, -7, -0.3};
const piqp_float dc[2] = {0.4, -0.1};
const piqp_float dA[2] = {0.2, 0.6};
const piqp_float db[1] = {-0.3};
const piqp_float dG[4] = {0.1, -0.2, 0.3, 0.4};
const piqp_float dh_l[2] = {0, 0.2};
const piqp_float dh_u[2] = {-0.4, 0};
const piqp_float dx_l[2] = {0, 0};
const piqp_float dx_u[2] = {0, 0.7};

using Mat = piqp::Mat<piqp_float>;
using SparseMat = piqp::SparseMat<piqp_float, piqp_int>;
using Vec = piqp::Vec<piqp_float>;

Mat mat(const piqp_float* data, piqp_int rows, piqp_int cols)
{
    return Eigen::Map<const Eigen::Matrix<piqp_float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, rows, cols);
}

Vec vec(const piqp_float* data, piqp_int size)
{
    return Eigen::Map<const Vec>(data, size);
}

piqp_settings settings(bool dense)
{
    piqp_settings settings;
    if (dense) piqp_set_default_settings_dense(&settings);
    else piqp_set_default_settings_sparse(&settings);
    settings.enable_derivatives = 1;
    return settings;
}

void setup_reference(piqp::DenseSolver<piqp_float>& solver)
{
    solver.settings().enable_derivatives = true;
    solver.setup(mat(P, n, n), vec(c, n), mat(A, p, n), vec(b, p), mat(G, m, n), vec(h_l, m), vec(h_u, m), vec(x_l, n), vec(x_u, n));
}

void setup_reference(piqp::SparseSolver<piqp_float, piqp_int>& solver)
{
    solver.settings().enable_derivatives = true;
    SparseMat P_sp = mat(P, n, n).sparseView();
    SparseMat A_sp = mat(A, p, n).sparseView();
    SparseMat G_sp = mat(G, m, n).sparseView();
    solver.setup(P_sp, vec(c, n), A_sp, vec(b, p), G_sp, vec(h_l, m), vec(h_u, m), vec(x_l, n), vec(x_u, n));
}

template<typename Solver>
piqp::Status vjp_reference(Solver& solver, bool all_arguments, const piqp::optional<piqp::Result<piqp_float>>& solve_result)
{
    if (!all_arguments) return solver.vjp(vec(grad_x, n), piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                                          piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, solve_result);
    return solver.vjp(vec(grad_x, n), vec(grad_y, p), vec(grad_z_l, m), vec(grad_z_u, m), vec(grad_z_bl, n), vec(grad_z_bu, n),
                      vec(grad_s_l, m), vec(grad_s_u, m), vec(grad_s_bl, n), vec(grad_s_bu, n), solve_result);
}

piqp_status vjp(piqp_workspace* work, bool all_arguments, const piqp_result* solve_result)
{
    if (!all_arguments) return piqp_vjp(work, grad_x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, solve_result);
    return piqp_vjp(work, grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, solve_result);
}

Mat dense_dP() { return mat(dP, n, n); }
Mat dense_dA() { return mat(dA, p, n); }
Mat dense_dG() { return mat(dG, m, n); }

// the perturbations of the matrices in the matrix type of the solver
Mat solver_mat(const piqp::DenseSolver<piqp_float>&, const Mat& M) { return M; }
SparseMat solver_mat(const piqp::SparseSolver<piqp_float, piqp_int>&, const Mat& M) { return M.sparseView(); }

template<typename Solver>
piqp::Status jvp_reference(Solver& solver, bool all_arguments, const piqp::optional<piqp::Result<piqp_float>>& solve_result)
{
    if (!all_arguments) return solver.jvp(piqp::nullopt, vec(dc, n), piqp::nullopt, piqp::nullopt, piqp::nullopt,
                                          piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, solve_result);
    return solver.jvp(solver_mat(solver, dense_dP()), vec(dc, n), solver_mat(solver, dense_dA()), vec(db, p),
                      solver_mat(solver, dense_dG()), vec(dh_l, m), vec(dh_u, m), vec(dx_l, n), vec(dx_u, n), solve_result);
}

piqp_status jvp_dense(piqp_workspace* work, bool all_arguments, const piqp_result* solve_result)
{
    if (!all_arguments) return piqp_jvp_dense(work, NULL, dc, NULL, NULL, NULL, NULL, NULL, NULL, NULL, solve_result);
    return piqp_jvp_dense(work, dP, dc, dA, db, dG, dh_l, dh_u, dx_l, dx_u, solve_result);
}

// the sparse perturbations, with views as piqp_csc
struct SparsePerturbation
{
    SparseMat dP_sp = dense_dP().sparseView();
    SparseMat dA_sp = dense_dA().sparseView();
    SparseMat dG_sp = dense_dG().sparseView();
    piqp_csc dP_csc = csc(dP_sp);
    piqp_csc dA_csc = csc(dA_sp);
    piqp_csc dG_csc = csc(dG_sp);

    static piqp_csc csc(SparseMat& M)
    {
        M.makeCompressed();
        return {(piqp_int) M.rows(), (piqp_int) M.cols(), (piqp_int) M.nonZeros(), M.outerIndexPtr(), M.innerIndexPtr(), M.valuePtr()};
    }
};

piqp_status jvp_sparse(piqp_workspace* work, bool all_arguments, const piqp_result* solve_result)
{
    if (!all_arguments) return piqp_jvp_sparse(work, NULL, dc, NULL, NULL, NULL, NULL, NULL, NULL, NULL, solve_result);
    SparsePerturbation d;
    return piqp_jvp_sparse(work, &d.dP_csc, dc, &d.dA_csc, db, &d.dG_csc, dh_l, dh_u, dx_l, dx_u, solve_result);
}

void expect_equal(const piqp_float* g, const Vec& ref)
{
    for (piqp::isize i = 0; i < ref.rows(); i++) EXPECT_EQ(g[i], ref(i)) << i;
}

void expect_equal(const piqp_float* g, const Mat& ref)
{
    for (piqp::isize i = 0; i < ref.rows(); i++)
        for (piqp::isize j = 0; j < ref.cols(); j++) EXPECT_EQ(g[i * ref.cols() + j], ref(i, j)) << i << ", " << j;
}

void expect_equal(const piqp_csc* g, const SparseMat& ref)
{
    ASSERT_EQ(g->m, ref.rows());
    ASSERT_EQ(g->n, ref.cols());
    ASSERT_EQ(g->nnz, ref.nonZeros());
    for (piqp_int j = 0; j <= g->n; j++) EXPECT_EQ(g->p[j], ref.outerIndexPtr()[j]);
    for (piqp_int k = 0; k < g->nnz; k++)
    {
        EXPECT_EQ(g->i[k], ref.innerIndexPtr()[k]);
        EXPECT_EQ(g->x[k], ref.valuePtr()[k]);
    }
}

template<typename VJPResult>
void expect_equal(const piqp_vjp_result* g, const VJPResult& ref)
{
    EXPECT_EQ(g->status, (piqp_status) ref.status);
    expect_equal(g->c, ref.c);
    expect_equal(g->b, ref.b);
    expect_equal(g->h_l, ref.h_l);
    expect_equal(g->h_u, ref.h_u);
    expect_equal(g->x_l, ref.x_l);
    expect_equal(g->x_u, ref.x_u);
}
void expect_equal(const piqp_jvp_result* t, const piqp::JVPResult<piqp_float>& ref)
{
    EXPECT_EQ(t->status, (piqp_status) ref.status);
    expect_equal(t->x, ref.x);
    expect_equal(t->y, ref.y);
    expect_equal(t->z_l, ref.z_l);
    expect_equal(t->z_u, ref.z_u);
    expect_equal(t->z_bl, ref.z_bl);
    expect_equal(t->z_bu, ref.z_bu);
    expect_equal(t->s_l, ref.s_l);
    expect_equal(t->s_u, ref.s_u);
    expect_equal(t->s_bl, ref.s_bl);
    expect_equal(t->s_bu, ref.s_bu);
}
} // namespace vjp_qp

TEST(CInterfaceTest, DenseVJP)
{
    using namespace vjp_qp;
    piqp_data_dense data = {n, p, m, P, c, A, b, G, h_l, h_u, x_l, x_u};
    piqp_settings s = settings(true);
    piqp_workspace* work;
    piqp_setup_dense(&work, &data, &s);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);

    piqp::DenseSolver<piqp_float> ref;
    setup_reference(ref);
    ASSERT_EQ(ref.solve(), piqp::Status::PIQP_SOLVED);

    for (bool all_arguments : {true, false})
    {
        SCOPED_TRACE(all_arguments ? "all arguments" : "only grad_x");
        ASSERT_EQ(vjp(work, all_arguments, NULL), PIQP_SOLVED);
        ASSERT_EQ(vjp_reference(ref, all_arguments, piqp::nullopt), piqp::Status::PIQP_SOLVED);
        expect_equal(work->vjp_result, ref.vjp_result());
        expect_equal(work->vjp_result->P, ref.vjp_result().P);
        expect_equal(work->vjp_result->A, ref.vjp_result().A);
        expect_equal(work->vjp_result->G, ref.vjp_result().G);
        EXPECT_EQ(work->vjp_result->P_csc, nullptr);

        ASSERT_EQ(vjp(work, all_arguments, work->result), PIQP_SOLVED);
        ASSERT_EQ(vjp_reference(ref, all_arguments, ref.result()), piqp::Status::PIQP_SOLVED);
        expect_equal(work->vjp_result, ref.vjp_result());
        expect_equal(work->vjp_result->G, ref.vjp_result().G);
    }

    piqp_cleanup(work);
}

TEST(CInterfaceTest, SparseVJP)
{
    using namespace vjp_qp;
    piqp_int P_p[3] = {0, 1, 2};
    piqp_int P_i[2] = {0, 1};
    piqp_float P_x[2] = {6, 4};
    piqp_int A_p[3] = {0, 1, 2};
    piqp_int A_i[2] = {0, 0};
    piqp_float A_x[2] = {1, -2};
    piqp_int G_p[3] = {0, 2, 3};
    piqp_int G_i[3] = {0, 1, 0};
    piqp_float G_x[3] = {1, 1, 1};
    piqp_data_sparse data = {n, p, m, piqp_csc_matrix(n, n, 2, P_p, P_i, P_x), c, piqp_csc_matrix(p, n, 2, A_p, A_i, A_x), b,
                             piqp_csc_matrix(m, n, 3, G_p, G_i, G_x), h_l, h_u, x_l, x_u};
    piqp_settings s = settings(false);
    piqp_workspace* work;
    piqp_setup_sparse(&work, &data, &s);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);

    piqp::SparseSolver<piqp_float, piqp_int> ref;
    setup_reference(ref);
    ASSERT_EQ(ref.solve(), piqp::Status::PIQP_SOLVED);

    for (bool all_arguments : {true, false})
    {
        SCOPED_TRACE(all_arguments ? "all arguments" : "only grad_x");
        ASSERT_EQ(vjp(work, all_arguments, NULL), PIQP_SOLVED);
        ASSERT_EQ(vjp_reference(ref, all_arguments, piqp::nullopt), piqp::Status::PIQP_SOLVED);
        expect_equal(work->vjp_result, ref.vjp_result());
        expect_equal(work->vjp_result->P_csc, ref.vjp_result().P);
        expect_equal(work->vjp_result->A_csc, ref.vjp_result().A);
        expect_equal(work->vjp_result->G_csc, ref.vjp_result().G);
        EXPECT_EQ(work->vjp_result->P, nullptr);

        ASSERT_EQ(vjp(work, all_arguments, work->result), PIQP_SOLVED);
        ASSERT_EQ(vjp_reference(ref, all_arguments, ref.result()), piqp::Status::PIQP_SOLVED);
        expect_equal(work->vjp_result, ref.vjp_result());
        expect_equal(work->vjp_result->G_csc, ref.vjp_result().G);
    }

    piqp_cleanup(work);
    free(data.P);
    free(data.A);
    free(data.G);
}

TEST(CInterfaceTest, DenseJVP)
{
    using namespace vjp_qp;
    piqp_data_dense data = {n, p, m, P, c, A, b, G, h_l, h_u, x_l, x_u};
    piqp_settings s = settings(true);
    piqp_workspace* work;
    piqp_setup_dense(&work, &data, &s);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);

    piqp::DenseSolver<piqp_float> ref;
    setup_reference(ref);
    ASSERT_EQ(ref.solve(), piqp::Status::PIQP_SOLVED);

    for (bool all_arguments : {true, false})
    {
        SCOPED_TRACE(all_arguments ? "all arguments" : "only dc");
        ASSERT_EQ(jvp_dense(work, all_arguments, NULL), PIQP_SOLVED);
        ASSERT_EQ(jvp_reference(ref, all_arguments, piqp::nullopt), piqp::Status::PIQP_SOLVED);
        expect_equal(work->jvp_result, ref.jvp_result());

        ASSERT_EQ(jvp_dense(work, all_arguments, work->result), PIQP_SOLVED);
        ASSERT_EQ(jvp_reference(ref, all_arguments, ref.result()), piqp::Status::PIQP_SOLVED);
        expect_equal(work->jvp_result, ref.jvp_result());
    }

    // the sparse function is rejected on a dense workspace
    EXPECT_EQ(jvp_sparse(work, false, NULL), PIQP_INVALID_SETTINGS);

    piqp_cleanup(work);
}

TEST(CInterfaceTest, SparseJVP)
{
    using namespace vjp_qp;
    piqp::SparseSolver<piqp_float, piqp_int> ref;
    setup_reference(ref);
    ASSERT_EQ(ref.solve(), piqp::Status::PIQP_SOLVED);

    SparseMat P_sp = mat(P, n, n).sparseView();
    SparseMat A_sp = mat(A, p, n).sparseView();
    SparseMat G_sp = mat(G, m, n).sparseView();
    piqp_csc P_csc = SparsePerturbation::csc(P_sp);
    piqp_csc A_csc = SparsePerturbation::csc(A_sp);
    piqp_csc G_csc = SparsePerturbation::csc(G_sp);
    piqp_data_sparse data = {n, p, m, &P_csc, c, &A_csc, b, &G_csc, h_l, h_u, x_l, x_u};
    piqp_settings s = settings(false);
    piqp_workspace* work;
    piqp_setup_sparse(&work, &data, &s);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);

    for (bool all_arguments : {true, false})
    {
        SCOPED_TRACE(all_arguments ? "all arguments" : "only dc");
        ASSERT_EQ(jvp_sparse(work, all_arguments, NULL), PIQP_SOLVED);
        ASSERT_EQ(jvp_reference(ref, all_arguments, piqp::nullopt), piqp::Status::PIQP_SOLVED);
        expect_equal(work->jvp_result, ref.jvp_result());

        ASSERT_EQ(jvp_sparse(work, all_arguments, work->result), PIQP_SOLVED);
        ASSERT_EQ(jvp_reference(ref, all_arguments, ref.result()), piqp::Status::PIQP_SOLVED);
        expect_equal(work->jvp_result, ref.jvp_result());
    }

    // the dense function is rejected on a sparse workspace
    EXPECT_EQ(jvp_dense(work, false, NULL), PIQP_INVALID_SETTINGS);

    piqp_cleanup(work);
}

TEST(CInterfaceTest, DerivativeStatus)
{
    using namespace vjp_qp;
    piqp_data_dense data = {n, p, m, P, c, A, b, G, h_l, h_u, x_l, x_u};

    // not enabled
    piqp_settings s;
    piqp_set_default_settings_dense(&s);
    EXPECT_EQ(s.enable_derivatives, 0);
    piqp_workspace* work;
    piqp_setup_dense(&work, &data, &s);
    EXPECT_EQ(work->vjp_result, nullptr);
    EXPECT_EQ(work->jvp_result, nullptr);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    EXPECT_EQ(vjp(work, false, NULL), PIQP_INVALID_SETTINGS);
    EXPECT_EQ(jvp_dense(work, false, NULL), PIQP_INVALID_SETTINGS);
    piqp_cleanup(work);

    s = settings(true);
    piqp_setup_dense(&work, &data, &s);
    ASSERT_NE(work->vjp_result, nullptr);
    ASSERT_NE(work->jvp_result, nullptr);
    EXPECT_EQ(work->vjp_result->status, PIQP_UNSOLVED);
    EXPECT_EQ(work->jvp_result->status, PIQP_UNSOLVED);
    EXPECT_EQ(vjp(work, false, NULL), PIQP_UNSOLVED);
    EXPECT_EQ(work->vjp_result->status, PIQP_UNSOLVED);
    EXPECT_EQ(jvp_dense(work, false, NULL), PIQP_UNSOLVED);
    EXPECT_EQ(work->jvp_result->status, PIQP_UNSOLVED);

    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    EXPECT_EQ(vjp(work, false, NULL), PIQP_SOLVED);
    EXPECT_EQ(work->vjp_result->status, PIQP_SOLVED);
    EXPECT_EQ(jvp_dense(work, false, NULL), PIQP_SOLVED);
    EXPECT_EQ(work->jvp_result->status, PIQP_SOLVED);

    piqp_result invalid = *work->result;
    invalid.info.rho = 0;
    EXPECT_EQ(vjp(work, false, &invalid), PIQP_INVALID_SETTINGS);
    EXPECT_EQ(work->vjp_result->status, PIQP_INVALID_SETTINGS);
    EXPECT_EQ(jvp_dense(work, false, &invalid), PIQP_INVALID_SETTINGS);
    EXPECT_EQ(work->jvp_result->status, PIQP_INVALID_SETTINGS);

    // a new solve invalidates the derivatives of the previous solution
    EXPECT_EQ(vjp(work, false, NULL), PIQP_SOLVED);
    EXPECT_EQ(jvp_dense(work, false, NULL), PIQP_SOLVED);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    EXPECT_EQ(work->vjp_result->status, PIQP_UNSOLVED);
    EXPECT_EQ(work->jvp_result->status, PIQP_UNSOLVED);

    EXPECT_EQ(vjp(work, false, NULL), PIQP_SOLVED);
    EXPECT_EQ(jvp_dense(work, false, NULL), PIQP_SOLVED);
    piqp_update_dense(work, NULL, c, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    EXPECT_EQ(work->vjp_result->status, PIQP_UNSOLVED);
    EXPECT_EQ(work->jvp_result->status, PIQP_UNSOLVED);

    piqp_cleanup(work);
}
