// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_RANDOM_HPP
#define PIQP_UTILS_RANDOM_HPP

#include <random>

#include "piqp/typedefs.hpp"
#include "piqp/model.hpp"

// adapted from https://github.com/Simple-Robotics/proxsuite/blob/main/include/proxsuite/proxqp/utils/random_qp_problems.hpp

namespace piqp
{

namespace rand
{

std::mt19937 gen(42);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::normal_distribution<double> normal_dist;


template<typename T>
Vec<T> vector_rand(isize n)
{
    Vec<T> v(n);

    for (isize i = 0; i < n; i++) {
        v(i) = T(normal_dist(gen));
    }

    return v;
}

template<typename T, typename I>
SparseMat<T, I> sparse_matrix_rand(isize n, isize m, T p)
{
    SparseMat<T, I> A(n, m);

    for (isize i = 0; i < n; i++) {
        for (isize j = 0; j < m; ++j) {
            if (uniform_dist(gen) < p) {
                A.insert(i, j) = T(normal_dist(gen));
            }
        }
    }
    A.makeCompressed();
    return A;
}

template<typename T, typename I>
SparseMat<T, I> sparse_positive_definite_upper_triangular_rand(isize n, T p, T rho = T(1e-2))
{
    SparseMat<T, I> P(n, n);
    P.setZero();

    for (isize i = 0; i < n; i++) {
        for (isize j = i + 1; j < n; ++j) {
            if (uniform_dist(gen) < p) {
                T random = T(normal_dist(gen));
                P.insert(i, j) = random;
            }
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> P_dense = P.toDense();
    Vec<T> eig_h = P_dense.template selfadjointView<Eigen::Upper>().eigenvalues();
    T min = eig_h.minCoeff();
    for (isize i = 0; i < n; i++) {
        P.coeffRef(i, i) += (rho + abs(min));
    }

    P.makeCompressed();
    return P;
}

template<typename T, typename I>
Model<T, I> sparse_strongly_convex_qp(isize dim, isize n_eq, isize n_ineq,
                                      T sparsity_factor, T strong_convexity_factor = T(1e-2))
{
    SparseMat<T, I> P = sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor, strong_convexity_factor);
    SparseMat<T, I> A = sparse_matrix_rand<T, I>(n_eq, dim, sparsity_factor);
    SparseMat<T, I> G = sparse_matrix_rand<T, I>(n_ineq, dim, sparsity_factor);

    Vec<T> x_sol = vector_rand<T>(dim);

    Vec<T> delta(n_ineq);
    delta.setZero();
    for (isize i = 0; i < n_ineq; i++) {
        // 30% of ineq constraints are active
        if (uniform_dist(gen) < 0.3) {
            delta(i) = uniform_dist(gen);
        }
    }

    Vec<T> c = vector_rand<T>(dim);
    Vec<T> b = A * x_sol;
    Vec<T> h = G * x_sol + delta;

    return Model<T, I>(P, c, A, b, G, h);
}

} // namespace rand

} // namespace piqp

#endif //PIQP_UTILS_RANDOM_HPP
