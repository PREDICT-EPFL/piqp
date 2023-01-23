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

namespace piqp
{

enum struct Status
{
    PIQP_SOLVED = 1,
    PIQP_MAX_ITER_REACHED = -1,
    PIQP_PRIMAL_INFEASIBLE = -2,
    PIQP_DUAL_INFEASIBLE = -3,
    PIQP_NUMERICS = -8,
    PIQP_UNSOLVED = -9,
    PIQP_INVALID_SETTINGS = -10
};

template<typename T>
struct Info
{
    Status status;

    isize iter;
    T factor_retires;
    T reg_limit;
    T no_primal_update; // dual infeasibility detection counter
    T no_dual_update;   // primal infeasibility detection counter

    T mu;
    T sigma;
    T primal_step;
    T dual_step;

    T primal_inf;
    T dual_inf;
};

template<typename T>
struct Result
{
    Vec<T> x;
    Vec<T> y;
    Vec<T> z;
    Vec<T> s;

    Vec<T> zeta;
    Vec<T> lambda;
    Vec<T> nu;

    Info<T> info;
};

} // namespace piqp

#endif //PIQP_RESULTS_HPP
