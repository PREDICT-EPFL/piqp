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
#include "piqp/dense/vjp_result.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/preconditioner.hpp"
#include "piqp/sparse/vjp_result.hpp"
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
    using VJPResultType = std::conditional_t<MatrixType == PIQP_DENSE, dense::VJPResult<T>, sparse::VJPResult<T, I>>;

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

    // differentiation state (vjp, jvp)
    Variables<T> m_linearization_point;  // point the gradients are computed at
    VJPResultType m_vjp_result;
    bool m_derivatives_enabled = false;     // differentiation workspace is allocated
    bool m_derivatives_ready = false;       // a valid solve of the current data is available
    bool m_derivatives_factor_valid = false; // the KKT is factored at the linearization point of the last solve() result

public:
    SolverBase();

    Settings<T>& settings() { return m_settings; }

    const Result<T>& result() const { return m_result; }

    Status solve();

    Status vjp(const optional<CVecRef<T>>& grad_x = nullopt,
                    const optional<CVecRef<T>>& grad_y = nullopt,
                    const optional<CVecRef<T>>& grad_z_l = nullopt,
                    const optional<CVecRef<T>>& grad_z_u = nullopt,
                    const optional<CVecRef<T>>& grad_z_bl = nullopt,
                    const optional<CVecRef<T>>& grad_z_bu = nullopt,
                    const optional<CVecRef<T>>& grad_s_l = nullopt,
                    const optional<CVecRef<T>>& grad_s_u = nullopt,
                    const optional<CVecRef<T>>& grad_s_bl = nullopt,
                    const optional<CVecRef<T>>& grad_s_bu = nullopt,
                    const optional<Result<T>>& solve_result = nullopt);

    const VJPResultType& vjp_result() const { return m_vjp_result; }

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

    bool factor_kkt(const Variables<T>& vars, Info<T>& info, bool* regularization_changed = nullptr);

    T calculate_mu();

    void calculate_step(T& alpha_s, T& alpha_z, const Variables<T>& vars);

    void update_residuals_nr(const Variables<T>& vars, Info<T>& info);

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

    void scale_results(Variables<T>& vars);

    void unscale_results(Variables<T>& vars);

    void pack_dual(Variables<T>& vars);

    void unpack_dual(Variables<T>& vars);

    void restore_dual();

    void reduce_dual(Variables<T>& vars);

    Status init_linearization_point(const Result<T>& solve_result);

    Status smooth_linearization_point(Info<T>& info);

    Status vjp_impl(const optional<CVecRef<T>>& grad_x,
                         const optional<CVecRef<T>>& grad_y,
                         const optional<CVecRef<T>>& grad_z_l,
                         const optional<CVecRef<T>>& grad_z_u,
                         const optional<CVecRef<T>>& grad_z_bl,
                         const optional<CVecRef<T>>& grad_z_bu,
                         const optional<CVecRef<T>>& grad_s_l,
                         const optional<CVecRef<T>>& grad_s_u,
                         const optional<CVecRef<T>>& grad_s_bl,
                         const optional<CVecRef<T>>& grad_s_bu,
                         const optional<Result<T>>& solve_result);

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
                              Variables<T>& rhs);

    void unscale_adjoint(Variables<T>& lambda);
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
