// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_LDLT_NO_PIVOT_TPP
#define PIQP_LDLT_NO_PIVOT_TPP

#include "piqp/common.hpp"
#include "piqp/dense/ldlt_no_pivot.hpp"

namespace piqp
{

namespace dense
{

extern template class LDLTNoPivot<common::Mat, Eigen::Lower>;

} // namespace dense

} // namespace piqp

#endif //PIQP_LDLT_NO_PIVOT_TPP
