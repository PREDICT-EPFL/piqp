// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_RANDOM_UTILS_HPP
#define PIQP_UTILS_RANDOM_UTILS_HPP

#include <random>

#include "piqp/typedefs.hpp"
#include "piqp/dense/model.hpp"
#include "piqp/sparse/model.hpp"

// adapted from https://github.com/Simple-Robotics/proxsuite/blob/main/include/proxsuite/proxqp/utils/random_qp_problems.hpp

namespace piqp
{

namespace rand
{

std::mt19937 gen(42);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::uniform_real_distribution<double> uniform_dist_pos(0.1, 100.0);
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

template<typename T>
Vec<T> vector_rand_strictly_positive(isize n)
{
    Vec<T> v(n);

    for (isize i = 0; i < n; i++) {
        v(i) = T(uniform_dist_pos(gen));
    }

    return v;
}

template<typename T>
Mat<T> dense_matrix_rand(isize n, isize m)
{
    Mat<T> A(n, m);
    for (isize i = 0; i < n; i++) {
        for (isize j = 0; j < m; ++j) {
            A(i, j) = T(normal_dist(gen));
        }
    }
    return A;
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

template<typename T>
Mat<T> dense_positive_definite_upper_triangular_rand(isize n, T rho = T(1e-2))
{
    Mat<T> mat = Mat<T>::Zero(n, n);
    for (isize i = 0; i < n; i++) {
        for (isize j = i + 1; j < n; ++j) {
            T random = T(normal_dist(gen));
            mat(i, j) = random;
        }
    }

    Vec<T> eig_h = mat.template selfadjointView<Eigen::Upper>().eigenvalues();
    T min = eig_h.minCoeff();
    for (isize i = 0; i < n; i++) {
        mat(i, i) += (rho + abs(min));
    }

    return mat;
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

template<typename T>
dense::Model<T> dense_strongly_convex_qp(isize dim, isize n_eq, isize n_ineq,
                                         T bounds_perc = T(0.5), T strong_convexity_factor = T(1e-2))
{
    Mat<T> P = dense_positive_definite_upper_triangular_rand<T>(dim, strong_convexity_factor);
    Mat<T> A = dense_matrix_rand<T>(n_eq, dim);
    Mat<T> G = dense_matrix_rand<T>(n_ineq, dim);

    Vec<T> x_sol = vector_rand<T>(dim);

    Vec<T> c = vector_rand<T>(dim);
    Vec<T> b(n_eq);
    if (n_eq > 0) {
        b = A * x_sol;
    }

    Vec<T> delta_u(n_ineq);
    Vec<T> delta_l(n_ineq);
    delta_u.setZero();
    delta_l.setZero();
    for (isize i = 0; i < n_ineq; i++) {
        // 30% of ineq constraints are inactive
        if (uniform_dist(gen) < 0.3) {
            delta_u(i) = uniform_dist(gen);
        }
        if (uniform_dist(gen) < 0.3) {
            delta_l(i) = uniform_dist(gen);
        }
    }

    Vec<T> h_l(n_ineq);
    Vec<T> h_u(n_ineq);
    if (n_ineq > 0) {
        h_l = G * x_sol - delta_l;
        h_u = G * x_sol + delta_u;
    }
    for (isize i = 0; i < n_ineq; i++) {
        double rand = uniform_dist(gen);
        // 33% only have lower bounds
        if (rand < 0.33) {
            h_l(i) = -std::numeric_limits<T>::infinity();
        }
        // 33% only have upper bounds
        else if (rand < 0.66) {
            h_u(i) = std::numeric_limits<T>::infinity();
        }
    }

    Vec<T> x_l = Vec<T>::Constant(dim, -std::numeric_limits<T>::infinity());
    Vec<T> x_u = Vec<T>::Constant(dim, std::numeric_limits<T>::infinity());
    for (isize i = 0; i < dim; i++) {
        double rand = uniform_dist(gen);
        if (rand < bounds_perc / 3) {
            x_l(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist(gen) < 0.5) {
                x_l(i) -= uniform_dist(gen);
            }
        }
        else if (rand < bounds_perc * 2 / 3) {
            x_u(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist(gen) < 0.5) {
                x_u(i) += uniform_dist(gen);
            }
        }
        else if (rand < bounds_perc) {
            x_l(i) = x_sol(i);
            x_u(i) = x_sol(i);
            if (uniform_dist(gen) < 0.5) {
                x_l(i) -= uniform_dist(gen);
            } else {
                x_u(i) += uniform_dist(gen);
            }
        }
    }

    return dense::Model<T>(P, c, A, b, G, h_l, h_u, x_l, x_u);
}

template<typename T, typename I>
sparse::Model<T, I> sparse_strongly_convex_qp(isize dim, isize n_eq, isize n_ineq, T sparsity_factor,
                                              T bounds_perc = T(0.5), T strong_convexity_factor = T(1e-2))
{
    SparseMat<T, I> P = sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor, strong_convexity_factor);
    SparseMat<T, I> A = sparse_matrix_rand<T, I>(n_eq, dim, sparsity_factor);
    SparseMat<T, I> G = sparse_matrix_rand<T, I>(n_ineq, dim, sparsity_factor);

    Vec<T> x_sol = vector_rand<T>(dim);

    Vec<T> delta_u(n_ineq);
    Vec<T> delta_l(n_ineq);
    delta_u.setZero();
    delta_l.setZero();
    for (isize i = 0; i < n_ineq; i++) {
        // 30% of ineq constraints are inactive
        if (uniform_dist(gen) < 0.3) {
            delta_u(i) = uniform_dist(gen);
        }
        if (uniform_dist(gen) < 0.3) {
            delta_l(i) = uniform_dist(gen);
        }
    }

    Vec<T> c = vector_rand<T>(dim);
    Vec<T> b = A * x_sol;

    Vec<T> h_l(n_ineq);
    Vec<T> h_u(n_ineq);
    if (n_ineq > 0) {
        h_l = G * x_sol - delta_l;
        h_u = G * x_sol + delta_u;
    }
    for (isize i = 0; i < n_ineq; i++) {
        double rand = uniform_dist(gen);
        // 33% only have lower bounds
        if (rand < 0.33) {
            h_l(i) = -std::numeric_limits<T>::infinity();
        }
        // 33% only have upper bounds
        else if (rand < 0.66) {
            h_u(i) = std::numeric_limits<T>::infinity();
        }
    }

    Vec<T> x_lb = Vec<T>::Constant(dim, -std::numeric_limits<T>::infinity());
    Vec<T> x_ub = Vec<T>::Constant(dim, std::numeric_limits<T>::infinity());
    for (isize i = 0; i < dim; i++) {
        double rand = uniform_dist(gen);
        if (rand < bounds_perc / 3) {
            x_lb(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist(gen) < 0.5) {
                x_lb(i) -= uniform_dist(gen);
            }
        }
        else if (rand < bounds_perc * 2 / 3) {
            x_ub(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist(gen) < 0.5) {
                x_ub(i) += uniform_dist(gen);
            }
        }
        // 1/6 of constraints are lower and upper bounded
        else if (rand < bounds_perc) {
            x_lb(i) = x_sol(i);
            x_ub(i) = x_sol(i);
            if (uniform_dist(gen) < 0.5) {
                x_lb(i) -= uniform_dist(gen);
            } else {
                x_ub(i) += uniform_dist(gen);
            }
        }
    }

    return sparse::Model<T, I>(P, c, A, b, G, h_l, h_u, x_lb, x_ub);
}

} // namespace rand

} // namespace piqp

#endif //PIQP_UTILS_RANDOM_UTILS_HPP
