// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SETTINGS_HPP
#define PIQP_SETTINGS_HPP

namespace piqp
{

template<typename T>
struct Settings
{
    T rho_init   = 1e-6;
    T delta_init = 1e-4;

    T feas_tol         = 1e-8;
    T dual_tol         = 1e-8;
    T reg_lower_limit  = 1e-10;

    isize max_iter           = 100;
    isize max_factor_retires = 10;

    T tau = 0.995;

    bool verbose = false;
};

} // namespace piqp

#endif //PIQP_SETTINGS_HPP
