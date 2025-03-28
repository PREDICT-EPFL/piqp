// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_SYSTEM_HPP
#define PIQP_KKT_SYSTEM_HPP

#include "piqp/kkt_solver_base.hpp"
#include "piqp/settings.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/kkt.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/kkt.hpp"
#ifdef PIQP_HAS_BLASFEO
#include "piqp/sparse/multistage_kkt.hpp"
#endif

namespace piqp
{

template<typename T, typename I, int MatrixType, int Mode>
class KKTSystem
{
protected:
	using DataType = typename std::conditional<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>::type;

	const DataType& data;
	const Settings<T>& settings;

	T m_delta;

	Vec<T> m_s;
	Vec<T> m_s_lb;
	Vec<T> m_s_ub;
	Vec<T> m_z_inv;
	Vec<T> m_z_lb_inv;
	Vec<T> m_z_ub_inv;

	// working variables
	Vec<T> work_x;
	Vec<T> work_z;

	std::unique_ptr<KKTSolverBase<T>> kkt_solver;

public:
	KKTSystem(const DataType& data, const Settings<T>& settings) : data(data), settings(settings) {}

	bool init()
	{
		m_s.resize(data.m);
		m_s_lb.resize(data.n);
		m_s_ub.resize(data.n);
		m_z_inv.resize(data.m);
		m_z_lb_inv.resize(data.n);
		m_z_ub_inv.resize(data.n);

		work_x.resize(data.n);
		work_z.resize(data.m);

		return init_kkt_solver<MatrixType>();
	}

	void update_data(int options) { kkt_solver->update_data(options); }

	bool update_scalings_and_factor(bool iterative_refinement,
									const T& rho, const T& delta,
                                    const CVecRef<T>& s, const CVecRef<T>& s_lb, const CVecRef<T>& s_ub,
                                    const CVecRef<T>& z, const CVecRef<T>& z_lb, const CVecRef<T>& z_ub)
    {
		Vec<T>& m_x_reg = work_x;
		Vec<T>& m_z_reg = work_z;

		m_delta = delta;
		m_s = s;
		m_s_lb = s_lb;
		m_s_ub = s_ub;
		m_z_inv.array() = z.array().inverse();
		m_z_lb_inv.array() = z_lb.array().inverse();
		m_z_ub_inv.array() = z_ub.array().inverse();

		m_x_reg.setConstant(rho);
		for (isize i = 0; i < data.n_lb; i++)
		{
			m_x_reg(data.x_lb_idx(i)) += data.x_b_scaling(i) * data.x_b_scaling(i) / (m_z_lb_inv(i) * m_s_lb(i) + m_delta);
		}
		for (isize i = 0; i < data.n_ub; i++)
		{
			m_x_reg(data.x_ub_idx(i)) += data.x_b_scaling(i) * data.x_b_scaling(i) / (m_z_ub_inv(i) * m_s_ub(i) + m_delta);
		}

		m_z_reg.array() = m_s.array() * m_z_inv.array() + delta;

		return kkt_solver->update_scalings_and_factor(delta, m_x_reg, m_z_reg);
    }

	void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y,
			   const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_z_lb, const CVecRef<T>& rhs_z_ub,
			   const CVecRef<T>& rhs_s, const CVecRef<T>& rhs_s_lb, const CVecRef<T>& rhs_s_ub,
			   VecRef<T> delta_x, VecRef<T> delta_y,
			   VecRef<T> delta_z, VecRef<T> delta_z_lb, VecRef<T> delta_z_ub,
			   VecRef<T> delta_s, VecRef<T> delta_s_lb, VecRef<T> delta_s_ub)
	{
		Vec<T>& rhs_x_bar = work_x;
		Vec<T>& rhs_z_bar = work_z;

		rhs_z_bar.array() = rhs_z.array() - m_z_inv.array() * rhs_s.array();

		rhs_x_bar = rhs_x;
		for (isize i = 0; i < data.n_lb; i++)
		{
			rhs_x_bar(data.x_lb_idx(i)) -= data.x_b_scaling(i) * (rhs_z_lb(i) - m_z_lb_inv(i) * rhs_s_lb(i))
				/ (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
		}
		for (isize i = 0; i < data.n_ub; i++)
		{
			rhs_x_bar(data.x_ub_idx(i)) += data.x_b_scaling(i) * (rhs_z_ub(i) - m_z_ub_inv(i) * rhs_s_ub(i))
				/ (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
		}

		kkt_solver->solve(rhs_x_bar, rhs_y, rhs_z_bar, delta_x, delta_y, delta_z);

		for (isize i = 0; i < data.n_lb; i++) {
			delta_z_lb(i) = (-data.x_b_scaling(i) * delta_x(data.x_lb_idx(i)) - rhs_z_lb(i) + m_z_lb_inv(i) * rhs_s_lb(i))
				/ (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
		}
		for (isize i = 0; i < data.n_ub; i++) {
			delta_z_ub(i) = (data.x_b_scaling(i) * delta_x(data.x_ub_idx(i)) - rhs_z_ub(i) + m_z_ub_inv(i) * rhs_s_ub(i))
				/ (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
		}

		delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());

		delta_s_lb.head(data.n_lb).array() = m_z_lb_inv.head(data.n_lb).array()
			* (rhs_s_lb.head(data.n_lb).array() - m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array());

		delta_s_ub.head(data.n_ub).array() = m_z_ub_inv.head(data.n_ub).array()
			* (rhs_s_ub.head(data.n_ub).array() - m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array());
	}

	// z = alpha * P * x
	void eval_P_x(const T& alpha, const CVecRef<T>& x, VecRef<T> z)
	{
		kkt_solver->eval_P_x(alpha, x, z);
	}

	// zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
	void eval_A_xn_and_AT_xt(const T& alpha_n, const T& alpha_t, const CVecRef<T>& xn, const CVecRef<T>& xt, VecRef<T> zn, VecRef<T> zt)
	{
		kkt_solver->eval_A_xn_and_AT_xt(alpha_n, alpha_t, xn, xt, zn, zt);
	}

	// zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
	void eval_G_xn_and_GT_xt(const T& alpha_n, const T& alpha_t, const CVecRef<T>& xn, const CVecRef<T>& xt, VecRef<T> zn, VecRef<T> zt)
	{
		kkt_solver->eval_G_xn_and_GT_xt(alpha_n, alpha_t, xn, xt, zn, zt);
	}

	void print_info() { kkt_solver->print_info(); }

protected:
	template<int MatrixTypeT = MatrixType>
	typename std::enable_if<MatrixTypeT == PIQP_DENSE, bool>::type
	init_kkt_solver()
	{
		switch (settings.kkt_solver) {
			case KKTSolver::dense_cholesky:
				kkt_solver = std::make_unique<dense::KKT<T>>(data, settings);
			break;
			default:
				piqp_eprint("kkt solver not supported\n");
			return false;
		}
		return true;
	}

	template<int MatrixTypeT = MatrixType>
	typename std::enable_if<MatrixTypeT == PIQP_SPARSE, bool>::type
	init_kkt_solver()
	{
		switch (settings.kkt_solver) {
			case KKTSolver::sparse_ldlt:
				kkt_solver = std::make_unique<sparse::KKT<T, I, Mode>>(data, settings);
				break;
#ifdef PIQP_HAS_BLASFEO
			case KKTSolver::sparse_multistage:
				kkt_solver = std::make_unique<sparse::MultistageKKT<T, I>>(data, settings);
				break;
#endif
			default:
				piqp_eprint("kkt solver not supported\n");
				return false;
		}
		return true;
	}
};

} // namespace piqp

#endif //PIQP_KKT_SYSTEM_HPP
