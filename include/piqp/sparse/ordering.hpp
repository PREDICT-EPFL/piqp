// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_ORDERING_HPP
#define PIQP_SPARSE_ORDERING_HPP

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace sparse
{

template<typename I>
class NaturalOrdering
{
public:
    template<typename T>
    void init(PIQP_MAYBE_UNUSED const SparseMat<T, I>& A) {}

    EIGEN_STRONG_INLINE I operator[](isize idx) const
    {
        return idx;
    }

    EIGEN_STRONG_INLINE I operator()(isize idx) const
    {
        return idx;
    }

    EIGEN_STRONG_INLINE I inv(isize idx) const
    {
        return I(idx);
    }

    template<typename T>
    void perm(VecRef<T> x, const CVecRef<T> b)
    {
        eigen_assert(x.rows() == b.rows() && "vector dimension missmatch!");
        x = b;
    }

    template<typename T>
    void permt(VecRef<T> x, const CVecRef<T> b)
    {
        eigen_assert(x.rows() == b.rows() && "vector dimension missmatch!");
        x = b;
    }
};

template<typename I>
class AMDOrdering
{
public:
    Vec<I> P;
    Vec<I> P_inv;

public:
    template<typename T>
    void init(const SparseMat<T, I>& A)
    {
        PIQP_TRACY_ZoneScopedN("piqp::AMDOrdering::init");

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, I> P_eigen;
        Eigen::AMDOrdering<I> amd_ordering;
        amd_ordering(A.template selfadjointView<Eigen::Upper>(), P_eigen);

        P = P_eigen.indices();

        isize n = P.rows();
        P_inv.resize(n);
        for (isize i = 0; i < n; i++)
        {
            P_inv[P[i]] = I(i);
        }
    }

    EIGEN_STRONG_INLINE I operator[](isize idx) const
    {
        return P[idx];
    }

    EIGEN_STRONG_INLINE I operator()(isize idx) const
    {
        return P[idx];
    }

    EIGEN_STRONG_INLINE I inv(isize idx) const
    {
        return P_inv[idx];
    }

    template<typename T>
    void perm(VecRef<T> x, const CVecRef<T>& b)
    {
        PIQP_TRACY_ZoneScopedN("piqp::AMDOrdering::perm");

        isize n = x.rows();
        eigen_assert(n == b.rows() && n == P.rows() && "vector dimension missmatch!");
        for (isize j = 0; j < n; j++)
        {
            x[j] = b[P[j]];
        }
    }

    template<typename T>
    void permt(VecRef<T> x, const CVecRef<T>& b)
    {
        PIQP_TRACY_ZoneScopedN("piqp::AMDOrdering::permt");

        isize n = x.rows();
        eigen_assert(n == b.rows() && n == P.rows() && "vector dimension missmatch!");
        for (isize j = 0 ; j < n; j++) {
            x[P[j]] = b[j];
        }
    }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/ordering.tpp"
#endif

#endif //PIQP_SPARSE_ORDERING_HPP
