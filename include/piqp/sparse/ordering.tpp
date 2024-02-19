// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_ORDERING_TPP
#define PIQP_SPARSE_ORDERING_TPP

#include "piqp/common.hpp"
#include "piqp/sparse/ordering.hpp"

namespace piqp
{

namespace sparse
{

extern template class AMDOrdering<common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_ORDERING_TPP
