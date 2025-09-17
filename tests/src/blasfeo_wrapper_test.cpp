#define PIQP_EIGEN_CHECK_MALLOC

#include <random>
#include "piqp/utils/random_utils.hpp"
#include "piqp/utils/blasfeo_wrapper.hpp"

#include "gtest/gtest.h"

using T = double;
using I = int;

using namespace piqp;
using namespace piqp::rand;


TEST(BlasfeoWrapperTest, blasfeo_dgemm) {
    std::uniform_int_distribution<int> uniform_dist_int(1, 100);

    size_t test_round = 100;
    // Test dgemm_nn
    // D <= beta * C + alpha * A * B
    for (size_t i = 0; i < test_round; i++) {
        T alpha = normal_dist(gen), beta = normal_dist(gen);
        I m = uniform_dist_int(gen), n = uniform_dist_int(gen), k = uniform_dist_int(gen);
        Mat<T> A_ref = dense_matrix_rand<T>(m, k);
        Mat<T> B_ref = dense_matrix_rand<T>(k, n);
        Mat<T> C_ref = dense_matrix_rand<T>(m, n);
        Mat<T> D1_ref(m, n);

        BlasfeoMat A(m, k), B(k, n), C(m, n), D(m, n);
        A.assign(A_ref);
        B.assign(B_ref);
        C.assign(C_ref);

        blasfeo_dgemm_nn(alpha, A, B, beta, C, D);
        D.load(D1_ref);
        ASSERT_TRUE(D1_ref.isApprox(alpha * A_ref * B_ref + beta * C_ref));
    }

    // Test dgemm_tn
    // D <= beta * C + alpha * A^T * B
    for (size_t i = 0; i < test_round; i++) {
        T alpha = normal_dist(gen), beta = normal_dist(gen);
        I m = uniform_dist_int(gen), n = uniform_dist_int(gen), k = uniform_dist_int(gen);
        Mat<T> AT_ref = dense_matrix_rand<T>(m, k);
        Mat<T> B_ref = dense_matrix_rand<T>(k, n);
        Mat<T> C_ref = dense_matrix_rand<T>(m, n);
        Mat<T> D_ref(m, n);

        BlasfeoMat A(k, m), B(k, n), C(m, n), D(m, n);
        A.assign(AT_ref.transpose());
        B.assign(B_ref);
        C.assign(C_ref);

        blasfeo_dgemm_tn(alpha, A, B, beta, C, D);
        D.load(D_ref);
        ASSERT_TRUE(D_ref.isApprox(alpha * AT_ref * B_ref + beta * C_ref));
    }

    // Test dgemm_tt
    // D <= beta * C + alpha * A^T * B^T
    for (size_t i = 0; i < test_round; i++) {
        T alpha = normal_dist(gen), beta = normal_dist(gen);
        I m = uniform_dist_int(gen), n = uniform_dist_int(gen), k = uniform_dist_int(gen);
        Mat<T> AT_ref = dense_matrix_rand<T>(m, k);
        Mat<T> BT_ref = dense_matrix_rand<T>(k, n);
        Mat<T> C_ref = dense_matrix_rand<T>(m, n);
        Mat<T> D1_ref(m, n);

        BlasfeoMat A(k, m), B(n, k), C(m, n), D1(m, n);
        A.assign(AT_ref.transpose());
        B.assign(BT_ref.transpose());
        C.assign(C_ref);

        blasfeo_dgemm_tt(alpha, A, B, beta, C, D1);
        D1.load(D1_ref);
        ASSERT_TRUE(D1_ref.isApprox(alpha * AT_ref * BT_ref + beta * C_ref));
    }
}

TEST(BlasfeoWrapperTest, blasfeo_dsyrk) {
    std::uniform_real_distribution<> dis_double(-100, 100.0);
    std::uniform_int_distribution<> dis_int(3, 100);

    size_t test_round = 100;
    for (size_t i = 0 ; i < test_round; i++) {
        T alpha = dis_double(gen), beta = dis_double(gen);
        I m = dis_int(gen), k = dis_int(gen);

        const Mat<T> A_ref = Mat<T>::Random(m, k), B_ref = Mat<T>::Random(m, k), C1_ref = Mat<T>::Random(m, m), C2_ref = Mat<T>::Random(k, k);
        Mat<T> D1_ref(m, m), D2_ref(k, k);

        BlasfeoMat A(m, k), B(m, k), C1(m, m), C2(k, k), D1(m, m), D2(k, k);
        A.assign(A_ref);
        B.assign(B_ref);
        C1.assign(C1_ref);
        C2.assign(C2_ref);

        // Test dsyrk_ln
        blasfeo_dsyrk_ln(alpha, A, B, beta, C1, D1);
        D1.load(D1_ref);
        auto res1 = D1_ref.triangularView<Eigen::Lower>().toDenseMatrix();
        auto res2 = (alpha * A_ref * B_ref.transpose() + beta * C1_ref).triangularView<Eigen::Lower>().toDenseMatrix();
        ASSERT_TRUE(res1.isApprox(res2));

        // Test dsyrk_lt
        blasfeo_dsyrk_lt(alpha, A, B, beta, C2, D2);
        D2.load(D2_ref);
        res1 = D2_ref.triangularView<Eigen::Lower>().toDenseMatrix();
        res2 = (alpha * A_ref.transpose() * B_ref + beta * C2_ref).triangularView<Eigen::Lower>().toDenseMatrix();
        ASSERT_TRUE(res1.isApprox(res2));
    }
}

TEST(BlasfeoWrapperTest, blasfeo_dtrsm) {
    std::uniform_real_distribution<> dis_double(-100, 100.0);
    std::uniform_int_distribution<> dis_int(5, 100);

    size_t test_round = 50;
    for (size_t i = 0 ; i < test_round; i++) {
        T alpha = dis_double(gen);
        I m = dis_int(gen), k = dis_int(gen);

        Mat<T> A_ref = dense_positive_definite_upper_triangular_rand<T>(m).transpose();
        Mat<T> B_ref = dense_matrix_rand<T>(m, k);

        Mat<T> D1_ref(m, k), D2_ref(k, k);

        BlasfeoMat A(m, m), B(m, k), D1(m, k), D2(k, k);
        A.assign(A_ref);
        B.assign(B_ref);

        // D <= alpha * A^{-1} * B
        blasfeo_dtrsm_llnn(alpha, A, B, D1);
        D1.load(D1_ref);
        ASSERT_TRUE(D1_ref.isApprox(alpha * A_ref.inverse() * B_ref));

        // D <= alpha * A^{-T} * B
        blasfeo_dtrsm_lltn(alpha, A, B, D1);
        D1.load(D1_ref);
        ASSERT_TRUE(D1_ref.isApprox(alpha * A_ref.transpose().inverse() * B_ref));
    }
}


TEST(BlasfeoWrapperTest, blasfeo_dgemv) {
    std::uniform_int_distribution<int> uniform_dist_int(1, 100);
    size_t test_round = 100;
    for (size_t i = 0; i < test_round; i++) {
        double alpha = normal_dist(gen);
        double beta = normal_dist(gen);
        int m = uniform_dist_int(gen);
        int n = uniform_dist_int(gen);

        Mat<T> A_ref = dense_matrix_rand<T>(m, n);
        Mat<T> B_ref = dense_matrix_rand<T>(n, n);
        Vec<T> x_ref = vector_rand<T>(n);
        Vec<T> y_ref = vector_rand<T>(m);
        Vec<T> z1_ref = vector_rand<T>(m);
        Vec<T> z2_ref = vector_rand<T>(n);

        BlasfeoMat A(m, n), B(n, n), C(m, n);
        A.assign(A_ref);
        B.assign(B_ref);
        BlasfeoVec x(n), y(m), z1(m), z2(n);
        x.assign(x_ref);
        y.assign(y_ref);

        // z1 <= beta * y + alpha * A * x
        blasfeo_dgemv_n(alpha, A, x, beta, y, z1);
        z1.load(z1_ref);
        ASSERT_TRUE(z1_ref.isApprox(alpha * A_ref * x_ref + beta * y_ref));

        // z2 <= beta * x + alpha * A^T * y
        blasfeo_dgemv_t(alpha, A, y, beta, x, z2);
        z2.load(z2_ref);
        ASSERT_TRUE(z2_ref.isApprox(alpha * A_ref.transpose() * y_ref + beta * x_ref));
    }
}

TEST(BlasfeoWrapperTest, blasfeo_dtrsv) {
    std::uniform_real_distribution<> dis_double(-100, 100.0);
    std::uniform_int_distribution<> dis_int(5, 100);
    size_t test_round = 100;
    for (size_t i = 0 ; i < test_round; i++) {
        I n = dis_int(gen);
        Mat<T> A_ref = dense_positive_definite_upper_triangular_rand<T>(n, n).transpose();
        Vec<T> b_ref = dense_matrix_rand<T>(n, 1);
        Vec<T> d_ref(n);

        BlasfeoMat A(n, n);
        BlasfeoVec b(n), d(n);
        A.assign(A_ref);
        b.assign(b_ref);

        // d <= A^{-1} * b
        blasfeo_dtrsv_lnn(A, b, d);
        d.load(d_ref);
        ASSERT_TRUE(d_ref.isApprox(A_ref.inverse() * b_ref));
        // d <= A^{-T} * b
        blasfeo_dtrsv_ltn(A, b, d);
        d.load(d_ref);
        ASSERT_TRUE(d_ref.isApprox(A_ref.transpose().inverse() * b_ref));
    }
}

