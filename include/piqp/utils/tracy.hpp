// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_TRACY_HPP
#define PIQP_UTILS_TRACY_HPP

#ifdef PIQP_HAS_TRACY
#include "tracy/Tracy.hpp"

#define PIQP_TRACY_ZoneScoped ZoneScoped
#define PIQP_TRACY_ZoneScopedN(name) ZoneScopedN(name)

#define PIQP_TRACY_ZoneValue(value) ZoneValue(value)

#else

#define PIQP_TRACY_ZoneScoped
#define PIQP_TRACY_ZoneScopedN(name)
#define PIQP_TRACY_ZoneValue(value)

#endif

#endif //PIQP_UTILS_TRACY_HPP
