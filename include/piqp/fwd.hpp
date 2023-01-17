// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_FWD_HPP
#define PIQP_FWD_HPP

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

#endif //PIQP_FWD_HPP
