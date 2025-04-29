// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SETTINGS_HPP
#define PIQP_SETTINGS_HPP

#include <limits>
#include "piqp/typedefs.hpp"

namespace piqp
{

enum class KKTSolver
{
    dense_cholesky,
    sparse_ldlt,
    sparse_ldlt_eq_cond,
    sparse_ldlt_ineq_cond,
    sparse_ldlt_cond,
    sparse_multistage
};

constexpr const char* kkt_solver_to_string(KKTSolver kkt_solver)
{
    switch (kkt_solver)
    {
        case KKTSolver::dense_cholesky: return "dense_cholesky";
        case KKTSolver::sparse_ldlt: return "sparse_ldlt";
        case KKTSolver::sparse_ldlt_eq_cond: return "sparse_ldlt_eq_cond";
        case KKTSolver::sparse_ldlt_ineq_cond: return "sparse_ldlt_ineq_cond";
        case KKTSolver::sparse_ldlt_cond: return "sparse_ldlt_cond";
        case KKTSolver::sparse_multistage: return "sparse_multistage";
        default: return "unknown";
    }
}

template<typename T>
struct Settings
{
    T rho_init = 1e-6;
    T delta_init = 1e-4;

    T eps_abs = 1e-8;
    T eps_rel = 1e-9;

    bool check_duality_gap = true;
    T eps_duality_gap_abs = 1e-8;
    T eps_duality_gap_rel = 1e-9;

    T infeasibility_threshold = 0.9;

    T reg_lower_limit = 1e-10;
    T reg_finetune_lower_limit = 1e-13;
    isize reg_finetune_primal_update_threshold = 7;
    isize reg_finetune_dual_update_threshold = 7;

    isize max_iter = 250;
    isize max_factor_retires = 10;

    bool preconditioner_scale_cost = false;
    bool preconditioner_reuse_on_update = false;
    isize preconditioner_iter = 10;

    T tau = 0.99;

    KKTSolver kkt_solver = KKTSolver::dense_cholesky;

    bool iterative_refinement_always_enabled = false;
    T iterative_refinement_eps_abs = 1e-12;
    T iterative_refinement_eps_rel = 1e-12;
    isize iterative_refinement_max_iter = 10;
    T iterative_refinement_min_improvement_rate = 5.0;
    T iterative_refinement_static_regularization_eps = 1e-8;
    T iterative_refinement_static_regularization_rel = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();

    bool verbose = false;
    bool compute_timings = false;

    bool verify_settings() const noexcept
    {
        return rho_init > 0 &&
               delta_init > 0 &&
               eps_abs > 0 &&
               eps_rel >= 0 &&
               eps_duality_gap_abs > 0 &&
               eps_duality_gap_rel >= 0 &&
               infeasibility_threshold >= 0 &&
               reg_lower_limit > 0 &&
               reg_finetune_primal_update_threshold >= 0 &&
               reg_finetune_dual_update_threshold >= 0 &&
               max_iter > 0 &&
               max_factor_retires > 0 &&
               preconditioner_iter >= 0 &&
               tau > 0 && tau <= 1 &&
               iterative_refinement_eps_abs > 0 &&
               iterative_refinement_eps_rel >= 0 &&
               iterative_refinement_max_iter >= 0 &&
               iterative_refinement_min_improvement_rate >= 1.0 &&
               iterative_refinement_static_regularization_eps > 0 &&
               iterative_refinement_static_regularization_rel >= 0;
    }
};

} // namespace piqp

#endif //PIQP_SETTINGS_HPP
