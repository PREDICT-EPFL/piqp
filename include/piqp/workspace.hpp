// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_WORKSPACE_HPP
#define PIQP_WORKSPACE_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/data.hpp"
#include "piqp/kkt_condensed.hpp"

namespace piqp
{

template<typename T, typename I>
struct Workspace
{
    T delta;
    T rho;

    Data<T, I> data;
};

} // namespace piqp

#endif //PIQP_WORKSPACE_HPP
