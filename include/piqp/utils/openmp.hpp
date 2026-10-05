// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_OPENMP_HPP
#define PIQP_UTILS_OPENMP_HPP

#include <algorithm>

#include "piqp/typedefs.hpp"

#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif

namespace piqp
{

// Minimum number of elements each thread has to process in cheap elementwise
// loops (e.g. the step length computation) before parallelization pays off.
// Waking up sleeping OpenMP worker threads can cost 10-20us per thread, see
// benchmarks/src/openmp_step_benchmark.cpp to tune this on a given platform.
constexpr isize min_elementwise_elems_per_thread = 32768;

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

// Number of threads to use for an elementwise loop over n elements, capped such
// that each thread processes at least min_elementwise_elems_per_thread elements.
// A return value of 1 means the loop should run serially without a parallel region.
inline int resolve_elementwise_num_threads(isize num_threads, isize n)
{
    const isize max_useful_threads = n / min_elementwise_elems_per_thread;
    return static_cast<int>((std::max)(isize(1), (std::min)(isize(resolve_num_threads(num_threads)), max_useful_threads)));
}

} // namespace piqp

#endif //PIQP_UTILS_OPENMP_HPP
