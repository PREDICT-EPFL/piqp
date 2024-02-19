// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_DATA_TPP
#define PIQP_DENSE_DATA_TPP

#include "piqp/common.hpp"
#include "piqp/dense/data.hpp"

namespace piqp
{

namespace dense
{

extern template struct Data<common::Scalar>;

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_DATA_TPP
