// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_BLOCK_INFO_HPP
#define PIQP_SPARSE_BLOCKSPARSE_BLOCK_INFO_HPP

namespace piqp
{

namespace sparse
{

template<typename I>
struct BlockInfo
{
    I start;
    I diag_size;
    I off_diag_size;
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_BLOCKSPARSE_BLOCK_INFO_HPP
