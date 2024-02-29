// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_FILESYSTEM_HPP
#define PIQP_UTILS_FILESYSTEM_HPP

#include "piqp/fwd.hpp"

#ifdef PIQP_STD_FILESYSTEM
#include <filesystem>
#else
#include "piqp/utils/ghc_filesystem.hpp"
#endif

namespace piqp
{

#ifdef PIQP_STD_FILESYSTEM
#include <filesystem>
namespace fs = std::filesystem;
#else
#include "filesystem.hpp"
namespace fs = ghc::filesystem;
#endif

} // namespace piqp

#endif //PIQP_UTILS_FILESYSTEM_HPP
