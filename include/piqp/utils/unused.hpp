// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_UNUSED_HPP
#define PIQP_UTILS_UNUSED_HPP

namespace piqp
{

template<typename... Args>
inline void unused(const Args&... args)
{
    using expand = int[];
    (void) expand{0, ((void) args, 0)...};
}

} // namespace piqp

#endif //PIQP_UTILS_UNUSED_HPP
