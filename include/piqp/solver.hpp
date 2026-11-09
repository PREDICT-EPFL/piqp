// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SOLVER_HPP
#define PIQP_SOLVER_HPP

#include <memory>

#include "piqp/results.hpp"
#include "piqp/settings.hpp"
#include "piqp/variables.hpp"
#include "piqp/kkt_system.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/preconditioner.hpp"
#include "piqp/dense/gradients.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/preconditioner.hpp"
#include "piqp/sparse/gradients.hpp"
#include "piqp/utils/optional.hpp"

namespace piqp
{

namespace detail
{

template<typename T, typename I, typename Preconditioner, int MatrixType>
struct is_solver_instantiated : std::false_type {};

} // namespace detail

template<typename T, typename I, typename Preconditioner, int MatrixType>
class SolverBase
{
#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
    static_assert(detail::is_solver_instantiated<T, I, Preconditioner, MatrixType>::value,
                  "This type combination has no pre-compiled template instantiation. "
                  "Use the default types (double, int), link against piqp::piqp_header_only, "
                  "or build with BUILD_WITH_TEMPLATE_INSTANTIATION=OFF.");
#endif

protected:
    using DataType = std::conditional_t<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>;
    using CMatRefType = std::conditional_t<MatrixType == PIQP_DENSE, CMatRef<T>, CSparseMatRef<T, I>>;
    using GradientsType = std::conditional_t<MatrixType == PIQP_DENSE, dense::Gradients<T>, sparse::Gradients<T, I>>;

    Result<T> m_result;
    Settings<T> m_settings;
    DataType m_data;
    Preconditioner m_preconditioner;
    KKTSystem<T, I, MatrixType> m_kkt_system;

    bool m_first_run = true;
    bool m_setup_done = false;
    bool m_enable_iterative_refinement = false;

    BasicVariables<T> res_nr;     // non-regularized residuals
    Variables<T> res;             // residuals
    Variables<T> step;            // primal and dual steps
    BasicVariables<T> prox_vars;  // proximal variables (xi, lambda, nu)
    BasicVariables<T> delta_vars; // displacement from the proximal variables (infeasibility certificate candidates)

    // work vectors for the infeasibility certificate check
    Vec<T> cert_work_n_1;
    Vec<T> cert_work_n_2;
    Vec<T> cert_work_p;
    Vec<T> cert_work_m_1;
    Vec<T> cert_work_m_2;

    T cert_c_x = 0;
    T cert_dual_lin = 0;
    T prox_c_x = 0;
    T prox_dual_lin = 0;
    T delta_c_x = 0;
    T delta_dual_lin = 0;

    // backward pass (differentiation) state
    Variables<T> m_result_scaled;                // final iterate in scaled space and compressed box layout
    Variables<T> m_adjoint_rhs;                  // right-hand side of the adjoint (transposed KKT) system
    Variables<T> m_adjoint;                      // adjoint variables lambda (s-fields are not meaningful)
    Variables<T> m_smoothed_scaled;              // relaxed point (s * z = mu) in scaled space, compressed box layout
    Variables<T> m_smoothed;                     // relaxed point in user space
    Variables<T> m_linearization_point_scaled;   // forward result provided to backward() in scaled space, compressed box layout
    Variables<T> m_linearization_point;          // forward result provided to backward() in user space, zero on infinite bounds
    GradientsType m_gradients;
    bool m_backward_ready = false;               // a valid solve of the current data is available
    bool m_gradients_allocated = false;

public:
    SolverBase();

    Settings<T>& settings() { return m_settings; }

    const Result<T>& result() const { return m_result; }

    Status solve();

    /**
     * Computes the gradients of a scalar loss L with respect to the problem data
     * (P, c, A, b, G, h_l, h_u, x_l, x_u) given the gradients of L with respect to the
     * solution (x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu) by implicit differentiation
     * of the KKT conditions.
     *
     * Missing arguments are treated as zero. z_bl, z_bu, s_bl, s_bu are n-dimensional
     * (same layout as in Result). The results are available via gradients().
     *
     * Each call factorizes the KKT system at the linearization point once (or once per smoothing
     * iteration if settings.gradient_smoothing is enabled).
     *
     * If forward_result is nullopt, the gradients are computed at the solution of the last solve(), which
     * must have returned PIQP_SOLVED or PIQP_MAX_ITER_REACHED and must not be followed by an update().
     *
     * If forward_result is provided, the gradients are computed at its primal-dual point and regularization
     * (forward_result.info.rho, forward_result.info.delta) instead, e.g. a stored result of an earlier
     * solve() of the current problem data. It has to be the output of a solve() (status PIQP_SOLVED
     * or PIQP_MAX_ITER_REACHED, s and z strictly positive on finite bounds). Pass a stored
     * optional<Result<T>> to avoid copying the result.
     *
     * settings.enable_backward has to be set before setup(), which allocates the gradient storage,
     * such that backward() does not allocate.
     *
     * @return PIQP_SOLVED on success, PIQP_INVALID_SETTINGS if settings.enable_backward was not set
     *         before setup(), PIQP_UNSOLVED if no valid solution is available or the dimensions mismatch,
     *         PIQP_NUMERICS if the KKT factorization or solve failed.
     */
    Status backward(const optional<CVecRef<T>>& grad_x = nullopt,
                    const optional<CVecRef<T>>& grad_y = nullopt,
                    const optional<CVecRef<T>>& grad_z_l = nullopt,
                    const optional<CVecRef<T>>& grad_z_u = nullopt,
                    const optional<CVecRef<T>>& grad_z_bl = nullopt,
                    const optional<CVecRef<T>>& grad_z_bu = nullopt,
                    const optional<CVecRef<T>>& grad_s_l = nullopt,
                    const optional<CVecRef<T>>& grad_s_u = nullopt,
                    const optional<CVecRef<T>>& grad_s_bl = nullopt,
                    const optional<CVecRef<T>>& grad_s_bu = nullopt,
                    const optional<Result<T>>& forward_result = nullopt);

    const GradientsType& gradients() const { return m_gradients; }

protected:
    void setup_impl(const CMatRefType& P,
                    const CVecRef<T>& c,
                    const optional<CMatRefType>& A,
                    const optional<CVecRef<T>>& b,
                    const optional<CMatRefType>& G,
                    const optional<CVecRef<T>>& h_l,
                    const optional<CVecRef<T>>& h_u,
                    const optional<CVecRef<T>>& x_l,
                    const optional<CVecRef<T>>& x_u);

    void update_impl(const optional<CMatRefType>& P,
                     const optional<CVecRef<T>>& c,
                     const optional<CMatRefType>& A,
                     const optional<CVecRef<T>>& b,
                     const optional<CMatRefType>& G,
                     const optional<CVecRef<T>>& h_l,
                     const optional<CVecRef<T>>& h_u,
                     const optional<CVecRef<T>>& x_l,
                     const optional<CVecRef<T>>& x_u);

    bool update_P(const CMatRefType& P);

    bool update_A(const CMatRefType& A);

    bool update_G(const CMatRefType& G);

    void init_workspace();

    Status solve_impl();

    T calculate_mu();

    // step lengths to the boundary of the non-negative orthant for (s, z) of vars (m_result if nullptr)
    void calculate_step(T& alpha_s, T& alpha_z, const Variables<T>* vars = nullptr);

    void update_residuals_nr();

    void update_residuals_r();

    T primal_res_nr();

    T primal_res_r();

    T primal_prox_inf();

    T dual_res_nr();

    T dual_res_r();

    T dual_prox_inf();

    void update_primal_prox_center();

    void update_dual_prox_center();

    Status check_infeasibility_certificates();

    void store_primal_infeasibility_certificate(const T& norm);

    void store_dual_infeasibility_certificate(const T& norm);

    // unscales vars (m_result if nullptr) from scaled to user space, box variables in compressed layout
    void unscale_results(Variables<T>* vars = nullptr);

    void restore_dual();

    bool scale_backward_point(const Result<T>& forward_result);

    bool factor_backward_kkt(const Variables<T>& vars_scaled, const T& rho, const T& delta);

    Status run_gradient_smoothing(const Variables<T>& start_scaled, const T& rho, const T& delta);

    T smoothing_residual(const Variables<T>& vars, const T& mu);

    void assemble_adjoint_rhs(const optional<CVecRef<T>>& grad_x,
                              const optional<CVecRef<T>>& grad_y,
                              const optional<CVecRef<T>>& grad_z_l,
                              const optional<CVecRef<T>>& grad_z_u,
                              const optional<CVecRef<T>>& grad_z_bl,
                              const optional<CVecRef<T>>& grad_z_bu,
                              const optional<CVecRef<T>>& grad_s_l,
                              const optional<CVecRef<T>>& grad_s_u,
                              const optional<CVecRef<T>>& grad_s_bl,
                              const optional<CVecRef<T>>& grad_s_bu,
                              const Variables<T>& linearization_point_scaled);

    void unscale_adjoint();
};

template<typename T, typename Preconditioner = dense::RuizEquilibration<T>>
class DenseSolver : public SolverBase<T, int, Preconditioner, PIQP_DENSE>
{
public:
    void setup(const CMatRef<T>& P,
               const CVecRef<T>& c,
               const optional<CMatRef<T>>& A = nullopt,
               const optional<CVecRef<T>>& b = nullopt,
               const optional<CMatRef<T>>& G = nullopt,
               const optional<CVecRef<T>>& h_l = nullopt,
               const optional<CVecRef<T>>& h_u = nullopt,
               const optional<CVecRef<T>>& x_l = nullopt,
               const optional<CVecRef<T>>& x_u = nullopt);

    void update(const optional<CMatRef<T>>& P = nullopt,
                const optional<CVecRef<T>>& c = nullopt,
                const optional<CMatRef<T>>& A = nullopt,
                const optional<CVecRef<T>>& b = nullopt,
                const optional<CMatRef<T>>& G = nullopt,
                const optional<CVecRef<T>>& h_l = nullopt,
                const optional<CVecRef<T>>& h_u = nullopt,
                const optional<CVecRef<T>>& x_l = nullopt,
                const optional<CVecRef<T>>& x_u = nullopt);
};

template<typename T, typename I = int, typename Preconditioner = sparse::RuizEquilibration<T, I>>
class SparseSolver : public SolverBase<T, I, Preconditioner, PIQP_SPARSE>
{
public:

    void setup(const CSparseMatRef<T, I>& P,
               const CVecRef<T>& c,
               const optional<CSparseMatRef<T, I>>& A = nullopt,
               const optional<CVecRef<T>>& b = nullopt,
               const optional<CSparseMatRef<T, I>>& G = nullopt,
               const optional<CVecRef<T>>& h_l = nullopt,
               const optional<CVecRef<T>>& h_u = nullopt,
               const optional<CVecRef<T>>& x_l = nullopt,
               const optional<CVecRef<T>>& x_u = nullopt);

    void update(const optional<CSparseMatRef<T, I>>& P = nullopt,
                const optional<CVecRef<T>>& c = nullopt,
                const optional<CSparseMatRef<T, I>>& A = nullopt,
                const optional<CVecRef<T>>& b = nullopt,
                const optional<CSparseMatRef<T, I>>& G = nullopt,
                const optional<CVecRef<T>>& h_l = nullopt,
                const optional<CVecRef<T>>& h_u = nullopt,
                const optional<CVecRef<T>>& x_l = nullopt,
                const optional<CVecRef<T>>& x_u = nullopt);
};

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace detail
{

template<>
struct is_solver_instantiated<common::Scalar, common::StorageIndex, common::dense::Preconditioner, PIQP_DENSE> : std::true_type {};
template<>
struct is_solver_instantiated<common::Scalar, common::StorageIndex, common::sparse::Preconditioner, PIQP_SPARSE> : std::true_type {};
template<>
struct is_solver_instantiated<common::Scalar, common::StorageIndex, dense::IdentityPreconditioner<common::Scalar>, PIQP_DENSE> : std::true_type {};
template<>
struct is_solver_instantiated<common::Scalar, common::StorageIndex, sparse::IdentityPreconditioner<common::Scalar, common::StorageIndex>, PIQP_SPARSE> : std::true_type {};

} // namespace detail

extern template class SolverBase<common::Scalar, common::StorageIndex, common::dense::Preconditioner, PIQP_DENSE>;
extern template class SolverBase<common::Scalar, common::StorageIndex, common::sparse::Preconditioner, PIQP_SPARSE>;
extern template class SolverBase<common::Scalar, common::StorageIndex, dense::IdentityPreconditioner<common::Scalar>, PIQP_DENSE>;
extern template class SolverBase<common::Scalar, common::StorageIndex, sparse::IdentityPreconditioner<common::Scalar, common::StorageIndex>, PIQP_SPARSE>;

extern template class DenseSolver<common::Scalar>;
extern template class SparseSolver<common::Scalar>;
extern template class DenseSolver<common::Scalar, dense::IdentityPreconditioner<common::Scalar>>;
extern template class SparseSolver<common::Scalar, common::StorageIndex, sparse::IdentityPreconditioner<common::Scalar, common::StorageIndex>>;

} // namespace piqp

#else
#include "piqp/solver.tpp"
#endif

#endif //PIQP_SOLVER_HPP
