// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_FWD_HPP
#define PIQP_FWD_HPP

#if __cplusplus >= 201703L
#define PIQP_WITH_CPP_17
#endif
#if __cplusplus >= 201402L
#define PIQP_WITH_CPP_14
#endif

#if defined(PIQP_WITH_CPP_17)
#define PIQP_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#define PIQP_MAYBE_UNUSED
#else
#define PIQP_MAYBE_UNUSED __attribute__((__unused__))
#endif

#ifdef PIQP_EIGEN_CHECK_MALLOC
#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#define EIGEN_RUNTIME_NO_MALLOC
#endif
#endif

#include <Eigen/Core>

#ifdef PIQP_EIGEN_CHECK_MALLOC
#ifdef EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#undef EIGEN_RUNTIME_NO_MALLOC
#undef EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#endif
#endif

// Check memory allocation for Eigen
#ifdef PIQP_EIGEN_CHECK_MALLOC
#define PIQP_EIGEN_MALLOC(allowed) ::Eigen::internal::set_is_malloc_allowed(allowed)
#define PIQP_EIGEN_MALLOC_ALLOWED() PIQP_EIGEN_MALLOC(true)
#define PIQP_EIGEN_MALLOC_NOT_ALLOWED() PIQP_EIGEN_MALLOC(false)
#else
#define PIQP_EIGEN_MALLOC(allowed)
#define PIQP_EIGEN_MALLOC_ALLOWED()
#define PIQP_EIGEN_MALLOC_NOT_ALLOWED()
#endif

#define PIQP_INF 1e30

#ifdef MATLAB
#define piqp_print mexPrintf
#define piqp_eprint mexPrintf
#elif defined R_LANG
#include <R_ext/Print.h>
#define piqp_print Rprintf
#define piqp_eprint REprintf
#else
#define piqp_print printf
#define piqp_eprint(...) fprintf(stderr, __VA_ARGS__)
#endif

#endif //PIQP_FWD_HPP
