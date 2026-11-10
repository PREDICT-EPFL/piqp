// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_RESULTS_HPP
#define PIQP_RESULTS_HPP

#include "piqp/typedefs.hpp"
#include "piqp/variables.hpp"

namespace piqp
{

enum Status
{
    PIQP_SOLVED = 1,
    PIQP_MAX_ITER_REACHED = -1,
    PIQP_PRIMAL_INFEASIBLE = -2,
    PIQP_DUAL_INFEASIBLE = -3,
    PIQP_NUMERICS = -8,
    PIQP_UNSOLVED = -9,
    PIQP_INVALID_SETTINGS = -10
};

constexpr const char* status_to_string(Status status)
{
    switch (status)
    {
        case Status::PIQP_SOLVED: return "solved";
        case Status::PIQP_MAX_ITER_REACHED: return "max iterations reached";
        case Status::PIQP_PRIMAL_INFEASIBLE: return "primal infeasible";
        case Status::PIQP_DUAL_INFEASIBLE: return "dual infeasible";
        case Status::PIQP_NUMERICS: return "numerics issue";
        case Status::PIQP_UNSOLVED: return "unsolved";
        case Status::PIQP_INVALID_SETTINGS: return "invalid settings";
        default: return "unknown";
    }
}

template<typename T>
struct Info
{
    Status status;

    isize iter;
    T rho;
    T delta;
    T mu;
    T sigma;
    T primal_step;
    T dual_step;

    T primal_res;
    T primal_res_rel;
    T dual_res;
    T dual_res_rel;

    T primal_res_reg;
    T primal_res_reg_rel;
    T dual_res_reg;
    T dual_res_reg_rel;

    T primal_prox_inf;
    T dual_prox_inf;

    T prev_primal_res;
    T prev_dual_res;

    T primal_obj;
    T dual_obj;
    T duality_gap;
    T duality_gap_rel;

    isize factor_retires;
    T reg_limit;
    isize no_primal_update; // dual infeasibility detection counter
    isize no_dual_update;   // primal infeasibility detection counter

    T setup_time;
    T update_time;
    T solve_time;
    T kkt_factor_time;
    T kkt_solve_time;
    T run_time;
};

template<typename T>
struct Result : Variables<T>
{
    Info<T> info;
};

} // namespace piqp

#endif //PIQP_RESULTS_HPP
