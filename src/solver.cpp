// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "piqp/solver.hpp"

namespace piqp
{

template class SolverBase<common::Scalar, common::StorageIndex, common::dense::Preconditioner, PIQP_DENSE>;
template class SolverBase<common::Scalar, common::StorageIndex, common::sparse::Preconditioner, PIQP_SPARSE>;

template class DenseSolver<common::Scalar>;
template class SparseSolver<common::Scalar>;

} // namespace piqp
