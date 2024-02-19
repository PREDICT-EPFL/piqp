// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SOLVER_TPP
#define PIQP_SOLVER_TPP

#include "piqp/common.hpp"
#include "piqp/solver.hpp"

namespace piqp
{

extern template class SolverBase<DenseSolver<common::Scalar, common::dense::Preconditioner>, common::Scalar, common::StorageIndex, common::dense::Preconditioner, PIQP_DENSE, KKTMode::KKT_FULL>;
extern template class SolverBase<SparseSolver<common::Scalar, common::StorageIndex, KKTMode::KKT_FULL, common::sparse::Preconditioner>, common::Scalar, common::StorageIndex, common::sparse::Preconditioner, PIQP_SPARSE, KKTMode::KKT_FULL>;

extern template class DenseSolver<common::Scalar>;
extern template class SparseSolver<common::Scalar>;

} // namespace piqp

#endif //PIQP_SOLVER_TPP
