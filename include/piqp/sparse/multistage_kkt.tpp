// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_MULTISTAGE_KKT_TPP
#define PIQP_SPARSE_MULTISTAGE_KKT_TPP

#include "piqp/common.hpp"
#include "piqp/sparse/multistage_kkt.hpp"

namespace piqp
{

namespace sparse
{

extern template class MultistageKKT<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_MULTISTAGE_KKT_TPP
