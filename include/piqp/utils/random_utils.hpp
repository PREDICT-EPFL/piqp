// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_RANDOM_UTILS_HPP
#define PIQP_UTILS_RANDOM_UTILS_HPP

#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "piqp/typedefs.hpp"
#include "piqp/variables.hpp"
#include "piqp/dense/model.hpp"
#include "piqp/sparse/model.hpp"

// adapted from https://github.com/Simple-Robotics/proxsuite/blob/main/include/proxsuite/proxqp/utils/random_qp_problems.hpp

namespace piqp
{

namespace rand
{

// function-local statics, such that the header can be included in multiple translation units
inline std::mt19937& gen() { static std::mt19937 g(42); return g; }
inline std::uniform_real_distribution<double>& uniform_dist() { static std::uniform_real_distribution<double> d(0.0, 1.0); return d; }
inline std::uniform_real_distribution<double>& uniform_dist_pos() { static std::uniform_real_distribution<double> d(0.1, 100.0); return d; }
inline std::normal_distribution<double>& normal_dist() { static std::normal_distribution<double> d; return d; }

// reseeds the generator and resets the distributions (normal_distribution caches values),
// i.e., the generated values only depend on the seed
inline void seed(unsigned int s)
{
    gen().seed(s);
    uniform_dist().reset();
    uniform_dist_pos().reset();
    normal_dist().reset();
}


template<typename T>
Vec<T> vector_rand(isize n)
{
    Vec<T> v(n);

    for (isize i = 0; i < n; i++) {
        v(i) = T(normal_dist()(gen()));
    }

    return v;
}

template<typename T>
Vec<T> vector_rand_strictly_positive(isize n)
{
    Vec<T> v(n);

    for (isize i = 0; i < n; i++) {
        v(i) = T(uniform_dist_pos()(gen()));
    }

    return v;
}

template<typename T>
Variables<T> variables_rand(isize n, isize p, isize m)
{
    Variables<T> v;
    v.x = vector_rand<T>(n);
    v.y = vector_rand<T>(p);
    v.z_l = vector_rand<T>(m);
    v.z_u = vector_rand<T>(m);
    v.z_bl = vector_rand<T>(n);
    v.z_bu = vector_rand<T>(n);
    v.s_l = vector_rand<T>(m);
    v.s_u = vector_rand<T>(m);
    v.s_bl = vector_rand<T>(n);
    v.s_bu = vector_rand<T>(n);
    return v;
}

template<typename T>
Mat<T> dense_matrix_rand(isize n, isize m)
{
    Mat<T> A(n, m);
    for (isize i = 0; i < n; i++) {
        for (isize j = 0; j < m; ++j) {
            A(i, j) = T(normal_dist()(gen()));
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
            if (uniform_dist()(gen()) < p) {
                A.insert(i, j) = T(normal_dist()(gen()));
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
            T random = T(normal_dist()(gen()));
            mat(i, j) = random;
        }
    }

    Vec<T> eig_h = mat.template selfadjointView<Eigen::Upper>().eigenvalues();
    T min = eig_h.minCoeff();
    for (isize i = 0; i < n; i++) {
        mat(i, i) += (rho + std::abs(min));
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
            if (uniform_dist()(gen()) < p) {
                T random = T(normal_dist()(gen()));
                P.insert(i, j) = random;
            }
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> P_dense = P.toDense();
    Vec<T> eig_h = P_dense.template selfadjointView<Eigen::Upper>().eigenvalues();
    T min = eig_h.minCoeff();
    for (isize i = 0; i < n; i++) {
        P.coeffRef(i, i) += (rho + std::abs(min));
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
        if (uniform_dist()(gen()) < 0.3) {
            delta_u(i) = uniform_dist()(gen());
        }
        if (uniform_dist()(gen()) < 0.3) {
            delta_l(i) = uniform_dist()(gen());
        }
    }

    Vec<T> h_l(n_ineq);
    Vec<T> h_u(n_ineq);
    if (n_ineq > 0) {
        h_l = G * x_sol - delta_l;
        h_u = G * x_sol + delta_u;
    }
    for (isize i = 0; i < n_ineq; i++) {
        double rand = uniform_dist()(gen());
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
        double rand = uniform_dist()(gen());
        if (rand < bounds_perc / 3) {
            x_l(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist()(gen()) < 0.5) {
                x_l(i) -= uniform_dist()(gen());
            }
        }
        else if (rand < bounds_perc * 2 / 3) {
            x_u(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist()(gen()) < 0.5) {
                x_u(i) += uniform_dist()(gen());
            }
        }
        else if (rand < bounds_perc) {
            x_l(i) = x_sol(i);
            x_u(i) = x_sol(i);
            if (uniform_dist()(gen()) < 0.5) {
                x_l(i) -= uniform_dist()(gen());
            } else {
                x_u(i) += uniform_dist()(gen());
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
        if (uniform_dist()(gen()) < 0.3) {
            delta_u(i) = uniform_dist()(gen());
        }
        if (uniform_dist()(gen()) < 0.3) {
            delta_l(i) = uniform_dist()(gen());
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
        double rand = uniform_dist()(gen());
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
        double rand = uniform_dist()(gen());
        if (rand < bounds_perc / 3) {
            x_lb(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist()(gen()) < 0.5) {
                x_lb(i) -= uniform_dist()(gen());
            }
        }
        else if (rand < bounds_perc * 2 / 3) {
            x_ub(i) = x_sol(i);
            // 50% of are inactive
            if (uniform_dist()(gen()) < 0.5) {
                x_ub(i) += uniform_dist()(gen());
            }
        }
        // 1/6 of constraints are lower and upper bounded
        else if (rand < bounds_perc) {
            x_lb(i) = x_sol(i);
            x_ub(i) = x_sol(i);
            if (uniform_dist()(gen()) < 0.5) {
                x_lb(i) -= uniform_dist()(gen());
            } else {
                x_ub(i) += uniform_dist()(gen());
            }
        }
    }

    return sparse::Model<T, I>(P, c, A, b, G, h_l, h_u, x_lb, x_ub);
}

/*
 * Indices of the active bounds in the user-space layout, i.e. h_l, h_u index rows of G
 * and x_l, x_u index variables.
 */
struct ActiveSet
{
    std::vector<isize> h_l;
    std::vector<isize> h_u;
    std::vector<isize> x_l;
    std::vector<isize> x_u;
};

// minimum relative norm of the part of an active constraint gradient orthogonal to the previous ones
constexpr double MIN_ACTIVE_RESIDUAL = 0.5;

/*
 * Builds a QP backwards from a prescribed KKT point (x, y, z) with a randomly sampled active set.
 * Each G row is disabled (10%), has only a lower (30%), only an upper (30%) or both bounds (30%);
 * a fraction bounds_perc of the variables gets box bounds in the same way. Every finite bound is active
 * with probability 0.5, at most one bound per row/variable, if its gradient has a relative part of at least
 * MIN_ACTIVE_RESIDUAL orthogonal to the rows of A and the previously activated gradients, which guarantees LICQ.
 * Active bounds get duals in [0.5, 1.5], inactive bounds slacks in [0.5, 1.5], such that the solution is strictly
 * complementary. A must have full row rank.
 */
template<typename Model, typename MatP, typename MatA, typename MatG>
std::pair<Model, ActiveSet> qp_with_known_active_set(const MatP& P, const MatA& A, const MatG& G, typename Model::T bounds_perc)
{
    using T = typename Model::T;
    const T inf = std::numeric_limits<T>::infinity();
    isize n = P.rows();
    isize p = A.rows();
    isize m = G.rows();
    Mat<T> A_dense = Mat<T>(A);
    Mat<T> G_dense = Mat<T>(G);
    auto uniform = []() { return T(uniform_dist()(gen())); };

    // finite bounds, and active bound: -1 lower, 1 upper, 0 none
    std::vector<bool> h_l_finite(m), h_u_finite(m), x_l_finite(n), x_u_finite(n);
    std::vector<int> G_active(m), x_active(n);
    // orthonormal basis of the rows of A and the active constraint gradients
    Mat<T> Q(n, 0);
    auto add_to_basis = [&](const Vec<T>& a, T min_residual) {
        Vec<T> r = a - Q * (Q.transpose() * a);
        if (!(r.norm() > min_residual * a.norm())) return false;
        Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);
        Q.col(Q.cols() - 1) = r / r.norm();
        return true;
    };
    for (isize i = 0; i < p; i++) {
        if (!add_to_basis(A_dense.row(i).transpose(), T(0))) throw std::invalid_argument("A must have full row rank");
    }
    auto sample_active = [&](bool lower_finite, bool upper_finite, const Vec<T>& gradient) {
        if (uniform() >= T(0.5)) return 0;
        if (!lower_finite && !upper_finite) return 0;
        if (!add_to_basis(gradient, T(MIN_ACTIVE_RESIDUAL))) return 0;
        if (lower_finite && (!upper_finite || uniform() < T(0.5))) return -1;
        return upper_finite ? 1 : 0;
    };
    for (isize i = 0; i < m; i++) {
        T u = uniform();
        h_l_finite[i] = (u >= T(0.1) && u < T(0.4)) || u >= T(0.7);
        h_u_finite[i] = u >= T(0.4);
        G_active[i] = sample_active(h_l_finite[i], h_u_finite[i], G_dense.row(i).transpose());
    }
    for (isize i = 0; i < n; i++) {
        T u = uniform();
        x_l_finite[i] = u < bounds_perc / 3 || (u >= 2 * bounds_perc / 3 && u < bounds_perc);
        x_u_finite[i] = u >= bounds_perc / 3 && u < bounds_perc;
        x_active[i] = sample_active(x_l_finite[i], x_u_finite[i], Vec<T>::Unit(n, i));
    }

    Vec<T> x = vector_rand<T>(n);
    Vec<T> y = vector_rand<T>(p);
    Vec<T> Gx = G_dense * x;
    auto margin = [&]() { return T(0.5) + uniform(); };

    ActiveSet active_set;
    Vec<T> h_l(m), h_u(m), z_l = Vec<T>::Zero(m), z_u = Vec<T>::Zero(m);
    for (isize i = 0; i < m; i++) {
        h_l(i) = h_l_finite[i] ? Gx(i) - margin() : -inf;
        h_u(i) = h_u_finite[i] ? Gx(i) + margin() : inf;
        if (G_active[i] < 0) { h_l(i) = Gx(i); z_l(i) = margin(); active_set.h_l.push_back(i); }
        if (G_active[i] > 0) { h_u(i) = Gx(i); z_u(i) = margin(); active_set.h_u.push_back(i); }
    }
    Vec<T> x_l(n), x_u(n), z_bl = Vec<T>::Zero(n), z_bu = Vec<T>::Zero(n);
    for (isize i = 0; i < n; i++) {
        x_l(i) = x_l_finite[i] ? x(i) - margin() : -inf;
        x_u(i) = x_u_finite[i] ? x(i) + margin() : inf;
        if (x_active[i] < 0) { x_l(i) = x(i); z_bl(i) = margin(); active_set.x_l.push_back(i); }
        if (x_active[i] > 0) { x_u(i) = x(i); z_bu(i) = margin(); active_set.x_u.push_back(i); }
    }

    // stationarity P x + c + A^T y + G^T (z_u - z_l) + z_bu - z_bl = 0 and primal feasibility A x = b
    Vec<T> c = -(P.template selfadjointView<Eigen::Upper>() * x + A_dense.transpose() * y
                 + G_dense.transpose() * (z_u - z_l) + z_bu - z_bl);
    Vec<T> b = A_dense * x;

    return std::make_pair(Model(P, c, A, b, G, h_l, h_u, x_l, x_u), active_set);
}

/*
 * Random strongly convex QPs with a known, strictly complementary solution satisfying LICQ (see qp_with_known_active_set).
 * A has a nonzero diagonal in its first n_eq columns, i.e., full row rank.
 */
template<typename T>
std::pair<dense::Model<T>, ActiveSet>
dense_strongly_convex_qp_with_known_active_set(isize dim, isize n_eq, isize n_ineq,
                                               T bounds_perc = T(0.5), T strong_convexity_factor = T(1e-2))
{
    Mat<T> P = dense_positive_definite_upper_triangular_rand<T>(dim, strong_convexity_factor);
    Mat<T> A = dense_matrix_rand<T>(n_eq, dim);
    Mat<T> G = dense_matrix_rand<T>(n_ineq, dim);
    for (isize i = 0; i < n_eq; i++) A(i, i) = T(1) + T(uniform_dist()(gen()));
    return qp_with_known_active_set<dense::Model<T>>(P, A, G, bounds_perc);
}

template<typename T, typename I>
std::pair<sparse::Model<T, I>, ActiveSet>
sparse_strongly_convex_qp_with_known_active_set(isize dim, isize n_eq, isize n_ineq, T sparsity_factor,
                                                T bounds_perc = T(0.5), T strong_convexity_factor = T(1e-2))
{
    SparseMat<T, I> P = sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor, strong_convexity_factor);
    SparseMat<T, I> A = sparse_matrix_rand<T, I>(n_eq, dim, sparsity_factor);
    SparseMat<T, I> G = sparse_matrix_rand<T, I>(n_ineq, dim, sparsity_factor);
    for (isize i = 0; i < n_eq; i++) A.coeffRef(i, i) = T(1) + T(uniform_dist()(gen()));
    A.makeCompressed();
    return qp_with_known_active_set<sparse::Model<T, I>>(P, A, G, bounds_perc);
}

} // namespace rand

} // namespace piqp

#endif //PIQP_UTILS_RANDOM_UTILS_HPP
