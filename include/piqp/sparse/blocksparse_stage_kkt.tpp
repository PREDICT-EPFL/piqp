// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_TPP
#define PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_TPP

#include "piqp/common.hpp"
#include "piqp/sparse/blocksparse_stage_kkt.hpp"

namespace piqp
{

namespace sparse
{

extern template class BlocksparseStageKKT<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_TPP
