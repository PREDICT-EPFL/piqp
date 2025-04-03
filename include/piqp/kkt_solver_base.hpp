// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_SOLVER_BASE_HPP
#define PIQP_KKT_SOLVER_BASE_HPP

#include "piqp/typedefs.hpp"

namespace piqp
{

template<typename T>
class KKTSolverBase
{
public:
    virtual ~KKTSolverBase() = default;

    virtual void update_data(int options) = 0;

    virtual bool update_scalings_and_factor(const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) = 0;

    virtual void solve(const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) = 0;

    // z = alpha * P * x
    virtual void eval_P_x(const T& alpha, const Vec<T>& x, Vec<T>& z) = 0;
    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    virtual void eval_A_xn_and_AT_xt(const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) = 0;
    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    virtual void eval_G_xn_and_GT_xt(const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) = 0;

    virtual void print_info() {};
};

} // namespace piqp

#endif //PIQP_KKT_SOLVER_BASE_HPP
