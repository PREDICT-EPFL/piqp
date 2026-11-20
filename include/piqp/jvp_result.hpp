// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_JVP_RESULT_HPP
#define PIQP_JVP_RESULT_HPP

#include "piqp/results.hpp"
#include "piqp/variables.hpp"

namespace piqp
{

/**
 * Tangent of the solution (x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu) along a perturbation of the
 * problem data, as computed by SolverBase::jvp(). The fields have the same dimensions as in Result.
 *
 * The tangents are only valid if status is PIQP_SOLVED. status is PIQP_UNSOLVED if jvp() was not
 * called yet or the problem changed since (setup(), update()), and the status returned by the last
 * jvp() otherwise.
 *
 * Entries of infinite bounds and of rows with both bounds infinite are zero.
 */
template<typename T>
struct JVPResult : Variables<T>
{
    Status status = Status::PIQP_UNSOLVED;
};

} // namespace piqp

#endif //PIQP_JVP_RESULT_HPP
