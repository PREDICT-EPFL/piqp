// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "piqp/sparse/ldlt.hpp"

namespace piqp
{

namespace sparse
{

template struct LDLt<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp
