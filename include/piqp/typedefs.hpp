// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TYPEDEFS_HPP
#define PIQP_TYPEDEFS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace piqp
{
namespace meta
{

template<typename T>
struct make_signed;
template<>
struct make_signed<unsigned char>
{
    using type = signed char;
};
template<>
struct make_signed<unsigned short>
{
    using type = signed short;
};
template<>
struct make_signed<unsigned int>
{
    using type = signed int;
};
template<>
struct make_signed<unsigned long>
{
    using type = signed long;
};
template<>
struct make_signed<unsigned long long>
{
    using type = signed long long;
};

} // namespace meta

using usize = decltype(sizeof(0));
using isize = meta::make_signed<usize>::type;

template<typename T, typename I>
using SparseMat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
template<typename T, typename I>
using SparseMatRef = Eigen::Ref<Eigen::SparseMatrix<T, Eigen::ColMajor, I>>;
template<typename T, typename I>
using CSparseMatRef = Eigen::Ref<const Eigen::SparseMatrix<T, Eigen::ColMajor, I>>;

template<typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template<typename T>
using CVecRef = Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template<typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template<typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template<typename T>
using CMatRef = Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

enum SolverMatrixType
{
    PIQP_DENSE = 0,
    PIQP_SPARSE = 1
};

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION

#define PIQP_CONCAT_IMPL(a, b) a##b
#define PIQP_CONCAT(a, b) PIQP_CONCAT_IMPL(a, b)
#define PIQP_EIGEN_ABI_CHECK_MAX_ALIGN PIQP_CONCAT(_piqp_eigen_abi_check_max_align_, EIGEN_MAX_ALIGN_BYTES)

extern void PIQP_EIGEN_ABI_CHECK_MAX_ALIGN();
inline void enforce_abi_compatibility() {
    PIQP_EIGEN_ABI_CHECK_MAX_ALIGN();
}

static const auto _abi_enforcer = (enforce_abi_compatibility(), 0);

#endif

} // namespace piqp

#endif //PIQP_TYPEDEFS_HPP
