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
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/preconditioner.hpp"
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

    Result<T> m_result;
    Settings<T> m_settings;
    DataType m_data;
    Preconditioner m_preconditioner;
    KKTSystem<T, I, MatrixType> m_kkt_system;

    bool m_first_run = true;
    bool m_setup_done = false;
    bool m_enable_iterative_refinement = false;
    bool m_warm_start_has_y = false;
    bool m_warm_start_has_z = false;
    bool m_warm_start_from_solve = false;

    BasicVariables<T> res_nr;    // non-regularized residuals
    Variables<T> res;            // residuals
    Variables<T> step;           // primal and dual steps
    BasicVariables<T> prox_vars; // proximal variables (xi, lambda, nu)

public:
    SolverBase();

    Settings<T>& settings() { return m_settings; }

    const Result<T>& result() const { return m_result; }

    void set_warm_start(const CVecRef<T>& x,
                        const optional<CVecRef<T>>& y = nullopt,
                        const optional<CVecRef<T>>& z_l = nullopt,
                        const optional<CVecRef<T>>& z_u = nullopt,
                        const optional<CVecRef<T>>& z_bl = nullopt,
                        const optional<CVecRef<T>>& z_bu = nullopt);

    Status solve();

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

    void calculate_step(T& alpha_s, T& alpha_z);

    void update_residuals_nr();

    void update_residuals_r();

    T primal_res_nr();

    T primal_res_r();

    T primal_prox_inf();

    T dual_res_nr();

    T dual_res_r();

    T dual_prox_inf();

    T init_compute_mu();

    void apply_smoothing(T sigma, T mu, const Variables<T>& s_kp1, const Variables<T>& z_k);

    Status init_cold_start();

    Status init_warm_start();

    Status init_from_guess(T sigma);

    void scale_results();

    void unscale_results();

    void pack_dual();

    void restore_dual();
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
