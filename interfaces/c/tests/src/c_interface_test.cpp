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
 * The backward() tests only check the C interface, i.e., the gradients have to be exactly the ones of the
 * C++ solver for the same arguments. QP with an active and an inactive inequality, such that the gradient
 * w.r.t. G is not symmetric and its layout is tested:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. x1 = 2 x2, x1 + x2 <= 0.3 (active), x1 >= -1, x2 <= 0.5
 */
namespace backward_qp
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
    settings.enable_backward = 1;
    return settings;
}

void setup_reference(piqp::DenseSolver<piqp_float>& solver)
{
    solver.settings().enable_backward = true;
    solver.setup(mat(P, n, n), vec(c, n), mat(A, p, n), vec(b, p), mat(G, m, n), vec(h_l, m), vec(h_u, m), vec(x_l, n), vec(x_u, n));
}

void setup_reference(piqp::SparseSolver<piqp_float, piqp_int>& solver)
{
    solver.settings().enable_backward = true;
    SparseMat P_sp = mat(P, n, n).sparseView();
    SparseMat A_sp = mat(A, p, n).sparseView();
    SparseMat G_sp = mat(G, m, n).sparseView();
    solver.setup(P_sp, vec(c, n), A_sp, vec(b, p), G_sp, vec(h_l, m), vec(h_u, m), vec(x_l, n), vec(x_u, n));
}

template<typename Solver>
piqp::Status backward_reference(Solver& solver, bool all_arguments, const piqp::optional<piqp::Result<piqp_float>>& forward_result)
{
    if (!all_arguments) return solver.backward(vec(grad_x, n), piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                                               piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, forward_result);
    return solver.backward(vec(grad_x, n), vec(grad_y, p), vec(grad_z_l, m), vec(grad_z_u, m), vec(grad_z_bl, n), vec(grad_z_bu, n),
                           vec(grad_s_l, m), vec(grad_s_u, m), vec(grad_s_bl, n), vec(grad_s_bu, n), forward_result);
}

piqp_status backward(piqp_workspace* work, bool all_arguments, const piqp_result* forward_result)
{
    if (!all_arguments) return piqp_backward(work, grad_x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, forward_result);
    return piqp_backward(work, grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu, grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result);
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

template<typename Gradients>
void expect_equal(const piqp_gradients* g, const Gradients& ref)
{
    EXPECT_EQ(g->status, (piqp_status) ref.status);
    expect_equal(g->c, ref.c);
    expect_equal(g->b, ref.b);
    expect_equal(g->h_l, ref.h_l);
    expect_equal(g->h_u, ref.h_u);
    expect_equal(g->x_l, ref.x_l);
    expect_equal(g->x_u, ref.x_u);
}
} // namespace backward_qp

TEST(CInterfaceTest, DenseBackward)
{
    using namespace backward_qp;
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
        ASSERT_EQ(backward(work, all_arguments, NULL), PIQP_SOLVED);
        ASSERT_EQ(backward_reference(ref, all_arguments, piqp::nullopt), piqp::Status::PIQP_SOLVED);
        expect_equal(work->gradients, ref.gradients());
        expect_equal(work->gradients->P, ref.gradients().P);
        expect_equal(work->gradients->A, ref.gradients().A);
        expect_equal(work->gradients->G, ref.gradients().G);
        EXPECT_EQ(work->gradients->P_csc, nullptr);

        ASSERT_EQ(backward(work, all_arguments, work->result), PIQP_SOLVED);
        ASSERT_EQ(backward_reference(ref, all_arguments, ref.result()), piqp::Status::PIQP_SOLVED);
        expect_equal(work->gradients, ref.gradients());
        expect_equal(work->gradients->G, ref.gradients().G);
    }

    piqp_cleanup(work);
}

TEST(CInterfaceTest, SparseBackward)
{
    using namespace backward_qp;
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
        ASSERT_EQ(backward(work, all_arguments, NULL), PIQP_SOLVED);
        ASSERT_EQ(backward_reference(ref, all_arguments, piqp::nullopt), piqp::Status::PIQP_SOLVED);
        expect_equal(work->gradients, ref.gradients());
        expect_equal(work->gradients->P_csc, ref.gradients().P);
        expect_equal(work->gradients->A_csc, ref.gradients().A);
        expect_equal(work->gradients->G_csc, ref.gradients().G);
        EXPECT_EQ(work->gradients->P, nullptr);

        ASSERT_EQ(backward(work, all_arguments, work->result), PIQP_SOLVED);
        ASSERT_EQ(backward_reference(ref, all_arguments, ref.result()), piqp::Status::PIQP_SOLVED);
        expect_equal(work->gradients, ref.gradients());
        expect_equal(work->gradients->G_csc, ref.gradients().G);
    }

    piqp_cleanup(work);
    free(data.P);
    free(data.A);
    free(data.G);
}

TEST(CInterfaceTest, BackwardStatus)
{
    using namespace backward_qp;
    piqp_data_dense data = {n, p, m, P, c, A, b, G, h_l, h_u, x_l, x_u};

    // not enabled
    piqp_settings s;
    piqp_set_default_settings_dense(&s);
    EXPECT_EQ(s.enable_backward, 0);
    piqp_workspace* work;
    piqp_setup_dense(&work, &data, &s);
    EXPECT_EQ(work->gradients, nullptr);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    EXPECT_EQ(backward(work, false, NULL), PIQP_INVALID_SETTINGS);
    piqp_cleanup(work);

    s = settings(true);
    piqp_setup_dense(&work, &data, &s);
    ASSERT_NE(work->gradients, nullptr);
    EXPECT_EQ(work->gradients->status, PIQP_UNSOLVED);
    EXPECT_EQ(backward(work, false, NULL), PIQP_UNSOLVED);
    EXPECT_EQ(work->gradients->status, PIQP_UNSOLVED);

    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    EXPECT_EQ(backward(work, false, NULL), PIQP_SOLVED);
    EXPECT_EQ(work->gradients->status, PIQP_SOLVED);

    piqp_result invalid = *work->result;
    invalid.info.rho = 0;
    EXPECT_EQ(backward(work, false, &invalid), PIQP_INVALID_SETTINGS);
    EXPECT_EQ(work->gradients->status, PIQP_INVALID_SETTINGS);

    EXPECT_EQ(backward(work, false, NULL), PIQP_SOLVED);
    piqp_update_dense(work, NULL, c, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    EXPECT_EQ(work->gradients->status, PIQP_UNSOLVED);

    piqp_cleanup(work);
}
