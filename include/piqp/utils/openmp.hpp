// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_OPENMP_HPP
#define PIQP_UTILS_OPENMP_HPP

#include "piqp/typedefs.hpp"

#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif

namespace piqp
{

inline int resolve_num_threads(isize num_threads)
{
#ifdef PIQP_HAS_OPENMP
    if (num_threads > 0) return static_cast<int>(num_threads);
    return omp_get_max_threads();
#else
    (void) num_threads;
    return 1;
#endif
}

} // namespace piqp

#endif //PIQP_UTILS_OPENMP_HPP
