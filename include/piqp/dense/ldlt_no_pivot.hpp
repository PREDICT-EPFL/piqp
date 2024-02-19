// This file is part of PIQP.
// It is a modification of LDLT.h and SimplicialCholesky.h which
// is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Keir Mierle <mierle@gmail.com>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2011 Timothy E. Holy <tim.holy@gmail.com >
// Copyright (C) 2011 Timothy E. Holy <tim.holy@gmail.com >
// Copyright (C) 2024 EPFL
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_LDLT_NO_PIVOT_HPP
#define PIQP_LDLT_NO_PIVOT_HPP

#include <iostream>
#include <Eigen/Dense>

namespace piqp
{

namespace dense
{

template<typename MatrixType, int UpLo = Eigen::Lower> class LDLTNoPivot;

} // namespace dense

} // namespace piqp

namespace Eigen
{

namespace internal
{

template<typename MatrixType_, int UpLo_>
struct traits<piqp::dense::LDLTNoPivot<MatrixType_, UpLo_>>
    : traits<MatrixType_>
{
    typedef MatrixXpr XprKind;
    typedef SolverStorage StorageKind;
    typedef int StorageIndex;
    enum
    {
        Flags = 0
    };
};

} // namespace internal

} // namespace Eigen

namespace piqp
{

namespace dense
{

namespace internal
{
template<typename MatrixType, int UpLo>
struct LDLTNoPivot_Traits;
}

/** \ingroup Cholesky_Module
  *
  * \class LDLTNoPivot
  *
  * \brief Robust Cholesky decomposition of a matrix without pivoting
  *
  * \tparam MatrixType_ the type of the matrix of which to compute the LDL^T Cholesky decomposition
  * \tparam UpLo_ the triangular part that will be used for the decomposition: Lower (default) or Upper.
  *             The other triangular part won't be read.
  *
  * Perform a robust Cholesky decomposition of a quasi-definite
  * matrix \f$ A \f$ such that \f$ A =  LDL^T \f$, where L
  * is lower triangular with a unit diagonal and D is a diagonal matrix.
  */
template<typename MatrixType_, int UpLo_> class LDLTNoPivot
    : public Eigen::SolverBase<LDLTNoPivot<MatrixType_, UpLo_> >
{
public:
    typedef MatrixType_ MatrixType;
    typedef Eigen::SolverBase<LDLTNoPivot> Base;
    friend class Eigen::SolverBase<LDLTNoPivot>;

    EIGEN_GENERIC_PUBLIC_INTERFACE(LDLTNoPivot)
    enum {
        MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef Eigen::Matrix<Scalar, RowsAtCompileTime, 1, 0, MaxRowsAtCompileTime, 1> TmpMatrixType;

    enum {
        PacketSize = Eigen::internal::packet_traits<Scalar>::size,
        AlignmentMask = int(PacketSize)-1,
        UpLo = UpLo_
    };

    typedef internal::LDLTNoPivot_Traits<MatrixType, UpLo> Traits;

    /**
      * \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via LDLTNoPivot::compute(const MatrixType&).
      */
    LDLTNoPivot()
        : m_matrix(),
          m_l1_norm(0),
          m_isInitialized(false),
          m_info(Eigen::NumericalIssue)
    {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa LDLTNoPivot()
      */
    explicit LDLTNoPivot(Eigen::Index size)
        : m_matrix(size, size),
          m_l1_norm(0),
          m_temporary(size),
          m_isInitialized(false),
          m_info(Eigen::NumericalIssue)
    {}

    template<typename InputType>
    explicit LDLTNoPivot(const Eigen::EigenBase<InputType>& matrix)
        : m_matrix(matrix.rows(), matrix.cols()),
          m_l1_norm(0),
          m_temporary(matrix.rows()),
          m_isInitialized(false),
          m_info(Eigen::NumericalIssue)
    {
        compute(matrix.derived());
    }

    /** \brief Constructs a LDLTNoPivot factorization from a given matrix
      *
      * This overloaded constructor is provided for \link InplaceDecomposition inplace decomposition \endlink when
      * \c MatrixType is a Eigen::Ref.
      *
      * \sa LDLTNoPivot(const EigenBase&)
      */
    template<typename InputType>
    explicit LDLTNoPivot(Eigen::EigenBase<InputType>& matrix)
        : m_matrix(matrix.derived()),
          m_l1_norm(0),
          m_temporary(matrix.rows()),
          m_isInitialized(false),
          m_info(Eigen::NumericalIssue)
    {
        compute(matrix.derived());
    }

    /** \returns a view of the upper triangular matrix U */
    inline typename Traits::MatrixU matrixU() const
    {
        eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
        return Traits::getU(m_matrix);
    }

    /** \returns a view of the lower triangular matrix L */
    inline typename Traits::MatrixL matrixL() const
    {
        eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
        return Traits::getL(m_matrix);
    }

    /** \returns the coefficients of the diagonal matrix D */
    inline Eigen::Diagonal<const MatrixType> vectorD() const
    {
        eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
        return m_matrix.diagonal();
    }

#ifdef EIGEN_PARSED_BY_DOXYGEN
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * Since this LDLTNoPivot class assumes anyway that the matrix A is invertible, the solution
      * theoretically exists and is unique regardless of b.
      *
      * \sa solveInPlace()
      */
    template<typename Rhs>
    inline const Solve<LDLTNoPivot, Rhs>
    solve(const MatrixBase<Rhs>& b) const;
#endif

    template<typename Derived>
    void solveInPlace(Eigen::MatrixBase<Derived> &bAndX) const;

    template<typename InputType>
    LDLTNoPivot& compute(const Eigen::EigenBase<InputType>& matrix);

    /** \returns an estimate of the reciprocal condition number of the matrix of
      *  which \c *this is the LDLT decomposition.
      */
    RealScalar rcond() const
    {
        eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
        return Eigen::internal::rcond_estimate_helper(m_l1_norm, *this);
    }

    /** \returns the LDLT decomposition matrix
      */
    inline const MatrixType& matrixLDLT() const
    {
        eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
        return m_matrix;
    }

    MatrixType reconstructedMatrix() const;


    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was successful,
      *          \c NumericalIssue if the matrix.appears not to be positive definite.
      */
    Eigen::ComputationInfo info() const
    {
        eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
        return m_info;
    }

    /** \returns the adjoint of \c *this, that is, a const reference to the decomposition itself as the underlying matrix is self-adjoint.
      *
      * This method is provided for compatibility with other matrix decompositions, thus enabling generic code such as:
      * \code x = decomposition.adjoint().solve(b) \endcode
      */
    const LDLTNoPivot& adjoint() const EIGEN_NOEXCEPT { return *this; }

    inline Eigen::Index rows() const EIGEN_NOEXCEPT { return m_matrix.rows(); }
    inline Eigen::Index cols() const EIGEN_NOEXCEPT { return m_matrix.cols(); }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename RhsType, typename DstType>
    void _solve_impl(const RhsType &rhs, DstType &dst) const;

    template<bool Conjugate, typename RhsType, typename DstType>
    void _solve_impl_transposed(const RhsType &rhs, DstType &dst) const;
#endif

protected:

    EIGEN_STATIC_ASSERT(!Eigen::NumTraits<Scalar>::IsInteger, THIS_FUNCTION_IS_NOT_FOR_INTEGER_NUMERIC_TYPES)

    /** \internal
      * Used to compute and store the Cholesky decomposition A = L D L^* = U^* D U.
      * The strict upper part is used during the decomposition, the strict lower
      * part correspond to the coefficients of L (its diagonal is equal to 1 and
      * is not stored), and the diagonal entries correspond to D.
      */
    MatrixType m_matrix;
    RealScalar m_l1_norm;
    TmpMatrixType m_temporary;
    bool m_isInitialized;
    Eigen::ComputationInfo m_info;
};

namespace internal {

template<int UpLo> struct ldlt_no_pivot_inplace;

template<> struct ldlt_no_pivot_inplace<Eigen::Lower>
{
    template<typename MatrixType, typename Workspace>
    static Eigen::Index unblocked(MatrixType& mat, Workspace& temp)
    {
        typedef typename MatrixType::RealScalar RealScalar;

        eigen_assert(mat.rows() == mat.cols());
        const Eigen::Index size = mat.rows();
        for (Eigen::Index k = 0; k < size; ++k)
        {
            // partition the matrix:
            //       A00 |  -  |  -
            // lu  = A10 | A11 |  -
            //       A20 | A21 | A22
            Eigen::Index rs = size - k - 1; // remaining size
            Eigen::Block<MatrixType, Eigen::Dynamic, 1> A21(mat, k + 1, k, rs, 1);
            Eigen::Block<MatrixType, 1, Eigen::Dynamic> A10(mat, k, 0, 1, k);
            Eigen::Block<MatrixType, Eigen::Dynamic, Eigen::Dynamic> A20(mat, k + 1, 0, rs, k);

            if (k > 0)
            {
                temp.head(k) = mat.diagonal().real().head(k).asDiagonal() * A10.adjoint();
                mat.coeffRef(k, k) -= (A10 * temp.head(k)).value();
                if (rs > 0)
                {
                    A21.noalias() -= A20 * temp.head(k);
                }
            }

            RealScalar x = Eigen::numext::real(mat.coeffRef(k, k));
            if (x == RealScalar(0)) return k;
            if (rs > 0) A21 /= x;
        }
        return -1;
    }

    template<typename MatrixType, typename Workspace>
    static Eigen::Index blocked(MatrixType& m, Workspace& temp)
    {
        eigen_assert(m.rows() == m.cols());
        Eigen::Index size = m.rows();
        if (size < 32)
            return unblocked(m, temp);

        Eigen::Index blockSize = size / 8;
        blockSize = (blockSize / 16) * 16;
        blockSize = (std::min)((std::max)(blockSize, Eigen::Index(8)), Eigen::Index(128));

        for (Eigen::Index k = 0; k < size; k += blockSize)
        {
            // partition the matrix:
            //       A00 |  -  |  -
            // lu  = A10 | A11 |  -
            //       A20 | A21 | A22
            Eigen::Index bs = (std::min)(blockSize, size - k);
            Eigen::Index rs = size - k - bs;
            Eigen::Block<MatrixType, Eigen::Dynamic, Eigen::Dynamic> A11(m, k, k, bs, bs);
            Eigen::Block<MatrixType, Eigen::Dynamic, Eigen::Dynamic> A21(m, k + bs, k, rs, bs);
            Eigen::Block<MatrixType, Eigen::Dynamic, Eigen::Dynamic> A22(m, k + bs, k + bs, rs, rs);

            // we use the unused upper triangular part as temporary storage
            Eigen::Block<MatrixType, Eigen::Dynamic, Eigen::Dynamic> A21_tmp(m, 0, size - bs, rs, bs);

            Eigen::Index ret;
            if ((ret = unblocked(A11, temp)) >= 0) return k + ret;
            if (rs > 0)
            {
                // A21 = A21 (A11^)^(-1) D11^(-1)
                A11.adjoint().template triangularView<Eigen::UnitUpper>().template solveInPlace<Eigen::OnTheRight>(A21);
                A21 = A21 * A11.diagonal().real().asDiagonal().inverse();

                // A22 -= A21 * D11 * A21^
                A21_tmp = A21 * A11.diagonal().real().asDiagonal();
                A22.template triangularView<Eigen::Lower>() -= A21_tmp * A21.transpose();
            }
        }
        return -1;
    }
};

template<> struct ldlt_no_pivot_inplace<Eigen::Upper>
{
    template<typename MatrixType, typename Workspace>
    static EIGEN_STRONG_INLINE Eigen::Index unblocked(MatrixType& mat, Workspace& temp)
    {
        Eigen::Transpose<MatrixType> matt(mat);
        return ldlt_no_pivot_inplace<Eigen::Lower>::unblocked(matt, temp);
    }
    template<typename MatrixType, typename Workspace>
    static EIGEN_STRONG_INLINE Eigen::Index blocked(MatrixType& mat, Workspace& temp)
    {
        Eigen::Transpose<MatrixType> matt(mat);
        return ldlt_no_pivot_inplace<Eigen::Lower>::blocked(matt, temp);
    }
};

template<typename MatrixType> struct LDLTNoPivot_Traits<MatrixType,Eigen::Lower>
{
    typedef const Eigen::TriangularView<const MatrixType, Eigen::UnitLower> MatrixL;
    typedef const Eigen::TriangularView<const typename MatrixType::AdjointReturnType, Eigen::UnitUpper> MatrixU;
    static inline MatrixL getL(const MatrixType& m) { return MatrixL(m); }
    static inline MatrixU getU(const MatrixType& m) { return MatrixU(m.adjoint()); }
};

template<typename MatrixType> struct LDLTNoPivot_Traits<MatrixType,Eigen::Upper>
{
    typedef const Eigen::TriangularView<const typename MatrixType::AdjointReturnType, Eigen::UnitLower> MatrixL;
    typedef const Eigen::TriangularView<const MatrixType, Eigen::UnitUpper> MatrixU;
    static inline MatrixL getL(const MatrixType& m) { return MatrixL(m.adjoint()); }
    static inline MatrixU getU(const MatrixType& m) { return MatrixU(m); }
};

} // end namespace internal

/** Compute / recompute the LDLT decomposition A = L D L^* = U^* D U of \a matrix
  */
template<typename MatrixType, int UpLo_>
template<typename InputType>
LDLTNoPivot<MatrixType,UpLo_>& LDLTNoPivot<MatrixType,UpLo_>::compute(const Eigen::EigenBase<InputType>& a)
{
    eigen_assert(a.rows() == a.cols());
    const Eigen::Index size = a.rows();
    m_matrix.resize(size, size);
    if (!Eigen::internal::is_same_dense(m_matrix, a.derived()))
        m_matrix = a.derived();

    // Compute matrix L1 norm = max abs column sum.
    m_l1_norm = RealScalar(0);
    // TODO move this code to SelfAdjointView
    for (Eigen::Index col = 0; col < size; ++col) {
        RealScalar abs_col_sum;
        if (UpLo_ == Eigen::Lower)
            abs_col_sum = m_matrix.col(col).tail(size - col).template lpNorm<1>() + m_matrix.row(col).head(col).template lpNorm<1>();
        else
            abs_col_sum = m_matrix.col(col).head(col).template lpNorm<1>() + m_matrix.row(col).tail(size - col).template lpNorm<1>();
        if (abs_col_sum > m_l1_norm)
            m_l1_norm = abs_col_sum;
    }

    m_temporary.resize(size);
    m_isInitialized = true;
    bool ok = internal::ldlt_no_pivot_inplace<UpLo>::blocked(m_matrix, m_temporary) == -1;
    m_info = ok ? Eigen::Success : Eigen::NumericalIssue;

    return *this;
}

#ifndef EIGEN_PARSED_BY_DOXYGEN
template<typename MatrixType_,int UpLo_>
template<typename RhsType, typename DstType>
void LDLTNoPivot<MatrixType_,UpLo_>::_solve_impl(const RhsType &rhs, DstType &dst) const
{
    _solve_impl_transposed<true>(rhs, dst);
}

template<typename MatrixType_,int UpLo_>
template<bool Conjugate, typename RhsType, typename DstType>
void LDLTNoPivot<MatrixType_,UpLo_>::_solve_impl_transposed(const RhsType &rhs, DstType &dst) const
{
    dst = rhs;

    if (!Conjugate)
    {
        matrixL().conjugate().solveInPlace(dst);
        dst.array() /= vectorD().real().array();
        matrixU().conjugate().solveInPlace(dst);
    }
    else
    {
        matrixL().solveInPlace(dst);
        dst.array() /= vectorD().real().array();
        matrixU().solveInPlace(dst);
    }
}
#endif

/** \internal use x = ldlt_no_pivot_object.solve(x);
  *
  * This is the \em in-place version of solve().
  *
  * \param bAndX represents both the right-hand side matrix b and result x.
  *
  * \returns true always! If you need to check for existence of solutions, use another decomposition like LU, QR, or SVD.
  *
  * This version avoids a copy when the right hand side matrix b is not
  * needed anymore.
  */
template<typename MatrixType, int UpLo_>
template<typename Derived>
void LDLTNoPivot<MatrixType,UpLo_>::solveInPlace(Eigen::MatrixBase<Derived> &bAndX) const
{
    eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
    eigen_assert(m_matrix.rows() == bAndX.rows());
    bAndX = this->solve(bAndX);
}

/** \returns the matrix represented by the decomposition,
 * i.e., it returns the product: L L^*.
 * This function is provided for debug purpose. */
template<typename MatrixType, int UpLo_>
MatrixType LDLTNoPivot<MatrixType,UpLo_>::reconstructedMatrix() const
{
    eigen_assert(m_isInitialized && "LDLTNoPivot is not initialized.");
    const Eigen::Index size = m_matrix.rows();
    MatrixType res(size,size);

    res.setIdentity();

    res = matrixU() * res;
    // D(L^)
    res = vectorD().real().asDiagonal() * res;
    // L(DL^)
    res = matrixL() * res;

    return res;
}

} // namespace dense

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/dense/ldlt_no_pivot.tpp"
#endif

#endif //PIQP_LDLT_NO_PIVOT_HPP
