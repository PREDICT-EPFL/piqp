// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_FWD_HPP
#define PIQP_KKT_FWD_HPP

namespace piqp
{

enum KKTMode
{
    FULL = 0,
    EQ_ELIMINATED = 0x1,
    INEQ_ELIMINATED = 0x2,
    ALL_ELIMINATED = EQ_ELIMINATED | INEQ_ELIMINATED
};

template<typename Derived, typename T, typename I, int Mode>
struct KKTImpl;

} // namespace piqp

#endif //PIQP_KKT_FWD_HPP
