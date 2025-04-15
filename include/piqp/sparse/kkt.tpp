// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_TPP
#define PIQP_SPARSE_KKT_TPP

#include "piqp/common.hpp"
#include "piqp/sparse/kkt.hpp"

namespace piqp
{

namespace sparse
{

extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_FULL, common::sparse::Ordering>;
extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_EQ_ELIMINATED, common::sparse::Ordering>;
extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_INEQ_ELIMINATED, common::sparse::Ordering>;
extern template class KKT<common::Scalar, common::StorageIndex, KKTMode::KKT_ALL_ELIMINATED, common::sparse::Ordering>;

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_TPP
