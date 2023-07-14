// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_HPP
#define PIQP_UTILS_HPP

#include <cstdio>

namespace piqp
{

inline void assert_exit(bool value, const char* message)
{
    if (!value)
    {
        fprintf(stderr, "%s", message);
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
}

} // namespace piqp

#endif //PIQP_UTILS_HPP
