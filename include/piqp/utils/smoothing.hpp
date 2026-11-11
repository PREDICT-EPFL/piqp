// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_SMOOTHING_HPP
#define PIQP_UTILS_SMOOTHING_HPP

#include <cmath>

namespace piqp
{

// Smoothing operator for the nonnegative orthant with log-barrier, including Moreau decomposition.
template<typename T>
void nonneg_smoothing(T sigma, T mu, T c, T& s, T& z)
{
    if (mu <= T(0)) {
        // Standard projection onto nonneg orthant
        s = (c > T(0)) ? c : T(0);
    } else {
        // Smoothing operator
        const T sigma_c = sigma * c;
        const T disc = sigma_c * sigma_c + T(4) * sigma * mu; // >= 0 always
        s = (sigma_c + std::sqrt(disc)) / (T(2) * sigma);
    }

    // Moreau decomposition
    // Satisfies z >= 0 and s*z = mu (approximate complementarity)
    z = sigma * (s - c);
}

} // namespace piqp

#endif // PIQP_UTILS_SMOOTHING_HPP
