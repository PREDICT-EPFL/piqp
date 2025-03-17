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
    KKT_FULL = 0,
    KKT_EQ_ELIMINATED = 0x1,
    KKT_INEQ_ELIMINATED = 0x2,
    KKT_ALL_ELIMINATED = KKT_EQ_ELIMINATED | KKT_INEQ_ELIMINATED
};

enum KKTUpdateOptions
{
    KKT_UPDATE_NONE = 0,
    KKT_UPDATE_P = 0x1,
    KKT_UPDATE_A = 0x2,
    KKT_UPDATE_G = 0x4
};

namespace sparse
{

template<typename Derived, typename T, typename I, int Mode>
class KKTImpl;

} // namespace sparse

} // namespace piqp

#endif //PIQP_KKT_FWD_HPP
