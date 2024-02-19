// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_COMMON_HPP
#define PIQP_COMMON_HPP

#include "piqp/typedefs.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
class RuizEquilibration;

} // namespace dense

namespace sparse
{

template<typename T, typename I>
class RuizEquilibration;

template<typename I>
class AMDOrdering;

} // namespace sparse

namespace common
{

using Scalar = double;
using StorageIndex = int;

using SparseMat = ::piqp::SparseMat<common::Scalar, common::StorageIndex>;
using Mat = ::piqp::Mat<common::Scalar>;

namespace dense
{

using Preconditioner = ::piqp::dense::RuizEquilibration<common::Scalar>;

} // namespace dense

namespace sparse
{

using Preconditioner = ::piqp::sparse::RuizEquilibration<common::Scalar, common::StorageIndex>;
using Ordering = ::piqp::sparse::AMDOrdering<StorageIndex>;

} // namespace sparse

} // namespace common

} // namespace piqp

#endif //PIQP_COMMON_HPP
