// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_PRECONDITIONER_HPP
#define PIQP_SPARSE_PRECONDITIONER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class RuizEquilibration
{
    static constexpr T min_scaling = 1e-4;
    static constexpr T max_scaling = 1e4;

    isize n = 0;
    isize p = 0;
    isize m = 0;

    T c = T(1);
    Vec<T> delta;
    Vec<T> delta_b;

    T c_inv = T(1);
    Vec<T> delta_inv;
    Vec<T> delta_b_inv;

    // temporary storage for the column norms of P in the cost scaling
    Vec<T> delta_cost;

public:
    void init(const Data<T, I>& data);

    void scale_data(Data<T, I>& data, bool reuse_prev_scaling = false, bool scale_cost = false, isize max_iter = 10, T epsilon = T(1e-3));

    void unscale_data(Data<T, I>& data);

    T scale_cost(T cost) const;

    T unscale_cost(T cost) const;

    template<typename Derived>
    auto scale_primal(const Eigen::MatrixBase<Derived>& x) const
    {
        return (x.array() * delta_inv.head(n).array()).matrix();
    }

    template<typename Derived>
    auto unscale_primal(const Eigen::MatrixBase<Derived>& x) const
    {
        return (x.array() * delta.head(n).array()).matrix();
    }

    template<typename Derived>
    auto scale_dual_eq(const Eigen::MatrixBase<Derived>& y) const
    {
        return (y.array() * c * delta_inv.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    auto unscale_dual_eq(const Eigen::MatrixBase<Derived>& y) const
    {
        return (y.array() * c_inv * delta.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    auto scale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const
    {
        return (z.array() * c * delta_inv.tail(m).array()).matrix();
    }

    template<typename Derived>
    auto unscale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const
    {
        return (z.array() * c_inv * delta.tail(m).array()).matrix();
    }

    template<typename Derived>
    auto scale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const
    {
        return (z_b.array() * c * delta_b_inv.array()).matrix();
    }

    template<typename Derived>
    auto unscale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const
    {
        return (z_b.array() * c_inv * delta_b.array()).matrix();
    }

    auto scale_dual_b_i(const T& z_b_i, Eigen::Index i) const
    {
        return z_b_i * c * delta_b_inv(i);
    }

    auto unscale_dual_b_i(const T& z_b_i, Eigen::Index i) const
    {
        return z_b_i * c_inv * delta_b(i);
    }

    template<typename Derived>
    auto scale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const
    {
        return (s.array() * delta.tail(m).array()).matrix();
    }

    template<typename Derived>
    auto unscale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const
    {
        return (s.array() * delta_inv.tail(m).array()).matrix();
    }

    template<typename Derived>
    auto scale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const
    {
        return (s_b.array() * delta_b.array()).matrix();
    }

    template<typename Derived>
    auto unscale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const
    {
        return (s_b.array() * delta_b_inv.array()).matrix();
    }

    auto scale_slack_b_i(const T& s_b_i, Eigen::Index i) const
    {
        return s_b_i * delta_b(i);
    }

    auto unscale_slack_b_i(const T& s_b_i, Eigen::Index i) const
    {
        return s_b_i * delta_b_inv(i);
    }

    template<typename Derived>
    auto scale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const
    {
        return (p_res_eq.array() * delta.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    auto unscale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const
    {
        return (p_res_eq.array() * delta_inv.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    auto scale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const
    {
        return (p_res_in.array() * delta.tail(m).array()).matrix();
    }

    template<typename Derived>
    auto unscale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const
    {
        return (p_res_in.array() * delta_inv.tail(m).array()).matrix();
    }

    template<typename Derived>
    auto scale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const
    {
        return (p_res_b.array() * delta_b.array()).matrix();
    }

    template<typename Derived>
    auto unscale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const
    {
        return (p_res_b.array() * delta_b_inv.array()).matrix();
    }

    auto scale_primal_res_b_i(const T& p_res_b_i, Eigen::Index i) const
    {
        return p_res_b_i * delta_b(i);
    }

    auto unscale_primal_res_b_i(const T& p_res_b_i, Eigen::Index i) const
    {
        return p_res_b_i * delta_b_inv(i);
    }

    template<typename Derived>
    auto scale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const
    {
        return (d_res.array() * c * delta.head(n).array()).matrix();
    }

    template<typename Derived>
    auto unscale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const
    {
        return (d_res.array() * c_inv * delta_inv.head(n).array()).matrix();
    }

protected:
    void limit_scaling(Vec<T>& d) const;
    void limit_scaling(T& d) const;
};

template<typename T, typename I>
class IdentityPreconditioner
{
public:
    void init(const Data<T, I>&) {}

    void scale_data(Data<T, I>&, bool = false, bool = false, isize = 0, T = T(0)) {}

    void unscale_data(Data<T, I>&) {}

    T scale_cost(T cost) const { return cost; }

    T unscale_cost(T cost) const { return cost; }

    template<typename Derived>
    auto& scale_primal(const Eigen::MatrixBase<Derived>& x) const { return x; }

    template<typename Derived>
    auto& unscale_primal(const Eigen::MatrixBase<Derived>& x) const { return x; }

    template<typename Derived>
    auto& scale_dual_eq(const Eigen::MatrixBase<Derived>& y) const { return y; }

    template<typename Derived>
    auto& unscale_dual_eq(const Eigen::MatrixBase<Derived>& y) const { return y; }

    template<typename Derived>
    auto& scale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const { return z; }

    template<typename Derived>
    auto& unscale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const { return z; }

    template<typename Derived>
    auto& scale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const { return z_b; }

    template<typename Derived>
    auto& unscale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const { return z_b; }

    auto& scale_dual_b_i(const T& z_b_i, Eigen::Index) const { return z_b_i; }

    auto& unscale_dual_b_i(const T& z_b_i, Eigen::Index) const { return z_b_i; }

    template<typename Derived>
    auto& scale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const { return s; }

    template<typename Derived>
    auto& unscale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const { return s; }

    template<typename Derived>
    auto& scale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const { return s_b; }

    template<typename Derived>
    auto& unscale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const { return s_b; }

    auto& scale_slack_b_i(const T& s_b_i, Eigen::Index) const { return s_b_i; }

    auto& unscale_slack_b_i(const T& s_b_i, Eigen::Index) const { return s_b_i; }

    template<typename Derived>
    auto& scale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const { return p_res_eq; }

    template<typename Derived>
    auto& unscale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const { return p_res_eq; }

    template<typename Derived>
    auto& scale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const { return p_res_in; }

    template<typename Derived>
    auto& unscale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const { return p_res_in; }

    template<typename Derived>
    auto& scale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const { return p_res_b; }

    template<typename Derived>
    auto& unscale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const { return p_res_b; }

    auto& scale_primal_res_b_i(const T& p_res_b_i, Eigen::Index) const { return p_res_b_i; }

    auto& unscale_primal_res_b_i(const T& p_res_b_i, Eigen::Index) const { return p_res_b_i; }

    template<typename Derived>
    auto& scale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const { return d_res; }

    template<typename Derived>
    auto& unscale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const { return d_res; }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace sparse
{

extern template class RuizEquilibration<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#else
#include "piqp/sparse/preconditioner.tpp"
#endif

#endif //PIQP_SPARSE_PRECONDITIONER_HPP
