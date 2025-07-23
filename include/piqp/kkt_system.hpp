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
#include "piqp/variables.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/kkt.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/kkt.hpp"
#ifdef PIQP_HAS_BLASFEO
#include "piqp/sparse/multistage_kkt.hpp"
#endif
#include "piqp/utils/tracy.hpp"

namespace piqp
{

template<typename T, typename I, int MatrixType>
class KKTSystem
{
protected:
	using DataType = std::conditional_t<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>;

	T m_rho;
	T m_delta;

	Vec<T> P_diag;

	Vec<T> m_s_l;
	Vec<T> m_s_u;
	Vec<T> m_s_bl;
	Vec<T> m_s_bu;
	Vec<T> m_z_l_inv;
	Vec<T> m_z_u_inv;
	Vec<T> m_z_bl_inv;
	Vec<T> m_z_bu_inv;

	Vec<T> m_x_reg;
	Vec<T> m_z_reg;

	Vec<T> rhs_x_bar;
	Vec<T> rhs_z_bar;

	// working variables
	Vec<T> work_x;
	Vec<T> work_z;

	// iterative refinement variables
	Vec<T> ref_err_x;
	Vec<T> ref_err_y;
	Vec<T> ref_err_z;
	Vec<T> ref_lhs_x;
	Vec<T> ref_lhs_y;
	Vec<T> ref_lhs_z;

	bool use_iterative_refinement = false;
	std::unique_ptr<KKTSolverBase<T, I, MatrixType>> kkt_solver;

public:
	KKTSystem() = default;

	KKTSystem(const KKTSystem& other) :
		m_rho(other.m_rho),
		m_delta(other.m_delta),
		P_diag(other.P_diag),
		m_s_l(other.m_s_l),
		m_s_u(other.m_s_u),
		m_s_bl(other.m_s_bl),
		m_s_bu(other.m_s_bu),
		m_z_l_inv(other.m_z_l_inv),
		m_z_u_inv(other.m_z_u_inv),
		m_z_bl_inv(other.m_z_bl_inv),
		m_z_bu_inv(other.m_z_bu_inv),
		m_x_reg(other.m_x_reg),
		m_z_reg(other.m_z_reg),
		rhs_x_bar(other.rhs_x_bar),
		rhs_z_bar(other.rhs_z_bar),
		work_x(other.work_x.rows()),
		work_z(other.work_z.rows()),
		ref_err_x(other.ref_err_x.rows()),
		ref_err_y(other.ref_err_y.rows()),
		ref_err_z(other.ref_err_z.rows()),
		ref_lhs_x(other.ref_lhs_x.rows()),
		ref_lhs_y(other.ref_lhs_y.rows()),
		ref_lhs_z(other.ref_lhs_z.rows()),
		use_iterative_refinement(other.use_iterative_refinement),
		kkt_solver(other.kkt_solver ? other.kkt_solver->clone() : nullptr) {}

	bool init(const DataType& data, const Settings<T>& settings)
	{
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::init");

		P_diag.resize(data.n);
		P_diag.setZero();

		m_s_l.resize(data.m);
		m_s_u.resize(data.m);
		m_s_bl.resize(data.n);
		m_s_bu.resize(data.n);
		m_z_l_inv.resize(data.m);
		m_z_u_inv.resize(data.m);
		m_z_bl_inv.resize(data.n);
		m_z_bu_inv.resize(data.n);

		m_x_reg.resize(data.n);
		m_z_reg.resize(data.m);

		rhs_x_bar.resize(data.n);
		rhs_z_bar.resize(data.m);

		work_x.resize(data.n);
		work_z.resize(data.m);

		ref_err_x.resize(data.n);
		ref_err_y.resize(data.p);
		ref_err_z.resize(data.m);
		ref_lhs_x.resize(data.n);
		ref_lhs_y.resize(data.p);
		ref_lhs_z.resize(data.m);

		extract_P_diag(data);

		return init_kkt_solver<MatrixType>(data, settings);
	}

	void update_data(const DataType& data, int options)
	{
		if (options & KKTUpdateOptions::KKT_UPDATE_P) {
			extract_P_diag(data);
		}

		kkt_solver->update_data(data, options);
	}

	bool update_scalings_and_factor(const DataType& data, const Settings<T>& settings, bool iterative_refinement,
		                            const T& rho, const T& delta, const Variables<T>& vars)
    {
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::update_scalings_and_factor");

		Vec<T>& m_z_reg_iter_ref = work_z;

		m_rho = rho;
		m_delta = delta;
		m_s_l = vars.s_l;
		m_s_u = vars.s_u;
		m_s_bl.head(data.n_x_l) = vars.s_bl.head(data.n_x_l);
		m_s_bu.head(data.n_x_u) = vars.s_bu.head(data.n_x_u);
		m_z_l_inv.array() = vars.z_l.array().inverse();
		m_z_u_inv.array() = vars.z_u.array().inverse();
		m_z_bl_inv.head(data.n_x_l).array() = vars.z_bl.head(data.n_x_l).array().inverse();
		m_z_bu_inv.head(data.n_x_u).array() = vars.z_bu.head(data.n_x_u).array().inverse();

		{
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::update_scalings_and_factor::x_reg");

			m_x_reg.setConstant(rho);
			for (isize i = 0; i < data.n_x_l; i++)
			{
				Eigen::Index idx = data.x_l_idx(i);
				m_x_reg(idx) += data.x_b_scaling(idx) * data.x_b_scaling(idx) / (m_z_bl_inv(i) * m_s_bl(i) + m_delta);
			}
			for (isize i = 0; i < data.n_x_u; i++)
			{
				Eigen::Index idx = data.x_u_idx(i);
				m_x_reg(idx) += data.x_b_scaling(idx) * data.x_b_scaling(idx) / (m_z_bu_inv(i) * m_s_bu(i) + m_delta);
			}
		}

		{
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::update_scalings_and_factor::z_reg");

			m_z_reg.setZero();
			for (isize i = 0; i < data.n_h_l; i++)
			{
				Eigen::Index idx = data.h_l_idx(i);
				m_z_reg(idx) += T(1) / (m_z_l_inv(idx) * m_s_l(idx) + delta);
			}
			for (isize i = 0; i < data.n_h_u; i++)
			{
				Eigen::Index idx = data.h_u_idx(i);
				m_z_reg(idx) += T(1) / (m_z_u_inv(idx) * m_s_u(idx) + delta);
			}
			m_z_reg.array() = m_z_reg.array().inverse();
			m_z_reg_iter_ref.array() = m_z_reg.array();
		}

		T delta_reg = delta;
		if (iterative_refinement)
		{
			T max_diag = (P_diag + m_x_reg).template lpNorm<Eigen::Infinity>();
			max_diag = (std::max)(max_diag, m_z_reg_iter_ref.template lpNorm<Eigen::Infinity>());

			T reg = settings.iterative_refinement_static_regularization_eps
				    + settings.iterative_refinement_static_regularization_rel * max_diag;

			delta_reg += reg;
			m_x_reg.array() += reg;
			m_z_reg_iter_ref.array() += reg;
		}

		use_iterative_refinement = iterative_refinement;
		return kkt_solver->update_scalings_and_factor(data, delta_reg, m_x_reg, m_z_reg_iter_ref);
    }

	bool solve(const DataType& data, const Settings<T>& settings, const Variables<T>& rhs, Variables<T>& lhs)
	{
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::solve");

		Vec<T>& lhs_z = work_z;

		{
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::solve::rhs_z_bar");

			rhs_z_bar.setZero();
			for (isize i = 0; i < data.n_h_l; i++)
			{
				Eigen::Index idx = data.h_l_idx(i);
				rhs_z_bar(idx) -= T(1) / (m_z_l_inv(idx) * m_s_l(idx) + m_delta) * (rhs.z_l(idx) - m_z_l_inv(idx) * rhs.s_l(idx));
			}
			for (isize i = 0; i < data.n_h_u; i++)
			{
				Eigen::Index idx = data.h_u_idx(i);
				rhs_z_bar(idx) += T(1) / (m_z_u_inv(idx) * m_s_u(idx) + m_delta) * (rhs.z_u(idx) - m_z_u_inv(idx) * rhs.s_u(idx));
			}
			rhs_z_bar.array() *= m_z_reg.array();
		}

		{
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::solve::rhs_x_bar");

			rhs_x_bar = rhs.x;
			for (isize i = 0; i < data.n_x_l; i++)
			{
				Eigen::Index idx = data.x_l_idx(i);
				rhs_x_bar(idx) -= data.x_b_scaling(idx) * (rhs.z_bl(i) - m_z_bl_inv(i) * rhs.s_bl(i))
					/ (m_s_bl(i) * m_z_bl_inv(i) + m_delta);
			}
			for (isize i = 0; i < data.n_x_u; i++)
			{
				Eigen::Index idx = data.x_u_idx(i);
				rhs_x_bar(idx) += data.x_b_scaling(idx) * (rhs.z_bu(i) - m_z_bu_inv(i) * rhs.s_bu(i))
					/ (m_s_bu(i) * m_z_bu_inv(i) + m_delta);
			}
		}

		kkt_solver->solve(data, rhs_x_bar, rhs.y, rhs_z_bar, lhs.x, lhs.y, lhs_z);

		if (use_iterative_refinement) {
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::solve::iterative_refinement");

			T rhs_norm = inf_norm(rhs_x_bar, rhs.y, rhs_z_bar);

			T refine_error = get_refine_error(data,
										    lhs.x, lhs.y, lhs_z,
				                            rhs_x_bar, rhs.y, rhs_z_bar,
				                            ref_err_x, ref_err_y, ref_err_z);

			if (!std::isfinite(refine_error)) return false;

			for (isize i = 0; i < settings.iterative_refinement_max_iter; i++)
			{
				if (refine_error <= settings.iterative_refinement_eps_abs + settings.iterative_refinement_eps_rel * rhs_norm) {
					break;
				}
				T prev_refine_error = refine_error;

				kkt_solver->solve(data, ref_err_x, ref_err_y, ref_err_z, ref_lhs_x, ref_lhs_y, ref_lhs_z);

				// use ref_lhs to store refined solution
				ref_lhs_x += lhs.x;
				ref_lhs_y += lhs.y;
				ref_lhs_z += lhs_z;

				refine_error = get_refine_error(data,
					                          ref_lhs_x, ref_lhs_y, ref_lhs_z,
											  rhs_x_bar, rhs.y, rhs_z_bar,
											  ref_err_x, ref_err_y, ref_err_z);

				if (!std::isfinite(refine_error)) return false;

				T improvement_rate = prev_refine_error / refine_error;
				if (improvement_rate < settings.iterative_refinement_min_improvement_rate) {
					if (improvement_rate > T(1)) {
						std::swap(lhs.x, ref_lhs_x);
						std::swap(lhs.y, ref_lhs_y);
						std::swap(lhs_z, ref_lhs_z);
					}
					break;
				}
				std::swap(lhs.x, ref_lhs_x);
				std::swap(lhs.y, ref_lhs_y);
				std::swap(lhs_z, ref_lhs_z);
			}

		} else {

			if (!lhs.x.allFinite() || !lhs.y.allFinite() || !lhs_z.allFinite()) {
				return false;
			}
		}

		{
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::solve::dual_recovery");

			isize i_l = 0;
			isize i_u = 0;
			for (isize i = 0; i < data.m; i++) {
				Eigen::Index idx_l = i_l < data.n_h_l ? data.h_l_idx(i_l) : -1;
				while (idx_l < i && i_l < data.n_h_l) { idx_l = data.h_l_idx(++i_l); }
				Eigen::Index idx_u = i_u < data.n_h_u ? data.h_u_idx(i_u) : -1;
				while (idx_u < i && i_u < data.n_h_u) { idx_u = data.h_u_idx(++i_u); }

				if (idx_l == i && idx_u == i) {
					T rz_l_bar = rhs.z_l(i) - m_z_l_inv(i) * rhs.s_l(i);
					T W_l_inv = T(1) / (m_z_l_inv(i) * m_s_l(i) + m_delta);
					T rz_u_bar = rhs.z_u(i) - m_z_u_inv(i) * rhs.s_u(i);
					T W_u_inv = T(1) / (m_z_u_inv(i) * m_s_u(i) + m_delta);
					T r_sum = W_l_inv * W_u_inv * (rz_l_bar + rz_u_bar);
					lhs.z_l(i) = -m_z_reg(i) * (r_sum + W_l_inv * lhs_z(i));
					lhs.z_u(i) = -m_z_reg(i) * (r_sum - W_u_inv * lhs_z(i));
					lhs.s_l(i) = m_z_l_inv(i) * (rhs.s_l(i) - m_s_l(i) * lhs.z_l(i));
					lhs.s_u(i) = m_z_u_inv(i) * (rhs.s_u(i) - m_s_u(i) * lhs.z_u(i));
				} else if (idx_l == i) {
					lhs.z_l(i) = -lhs_z(i);
					lhs.z_u(i) = T(0);
					lhs.s_l(i) = m_z_l_inv(i) * (rhs.s_l(i) - m_s_l(i) * lhs.z_l(i));
					lhs.s_u(i) = T(0);
				} else if (idx_u == i) {
					lhs.z_l(i) = T(0);
					lhs.z_u(i) = lhs_z(i);
					lhs.s_l(i) = T(0);
					lhs.s_u(i) = m_z_u_inv(i) * (rhs.s_u(i) - m_s_u(i) * lhs.z_u(i));
				} else {
					assert(false && "This should be unreachable...");
				}
			}
		}

		{
			PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::solve::box_dual_recovery");

			for (isize i = 0; i < data.n_x_l; i++) {
				Eigen::Index idx = data.x_l_idx(i);
				lhs.z_bl(i) = (-data.x_b_scaling(idx) * lhs.x(idx) - rhs.z_bl(i) + m_z_bl_inv(i) * rhs.s_bl(i))
					/ (m_s_bl(i) * m_z_bl_inv(i) + m_delta);
			}
			for (isize i = 0; i < data.n_x_u; i++) {
				Eigen::Index idx = data.x_u_idx(i);
				lhs.z_bu(i) = (data.x_b_scaling(idx) * lhs.x(idx) - rhs.z_bu(i) + m_z_bu_inv(i) * rhs.s_bu(i))
					/ (m_s_bu(i) * m_z_bu_inv(i) + m_delta);
			}

			lhs.s_bl.head(data.n_x_l).array() = m_z_bl_inv.head(data.n_x_l).array()
				* (rhs.s_bl.head(data.n_x_l).array() - m_s_bl.head(data.n_x_l).array() * lhs.z_bl.head(data.n_x_l).array());

			lhs.s_bu.head(data.n_x_u).array() = m_z_bu_inv.head(data.n_x_u).array()
				* (rhs.s_bu.head(data.n_x_u).array() - m_s_bu.head(data.n_x_u).array() * lhs.z_bu.head(data.n_x_u).array());
		}

		return true;
	}

	// z = alpha * P * x
	void eval_P_x(const DataType& data, const T& alpha, const Vec<T>& x, Vec<T>& z)
	{
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::eval_P_x");
		kkt_solver->eval_P_x(data, alpha, x, z);
	}

	// zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
	void eval_A_xn_and_AT_xt(const DataType& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
	{
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::eval_A_xn_and_AT_xt");
		kkt_solver->eval_A_xn_and_AT_xt(data, alpha_n, alpha_t, xn, xt, zn, zt);
	}

	// zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
	void eval_G_xn_and_GT_xt(const DataType& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
	{
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::eval_G_xn_and_GT_xt");
		kkt_solver->eval_G_xn_and_GT_xt(data, alpha_n, alpha_t, xn, xt, zn, zt);
	}

	void mul(const DataType& data, const Variables<T>& lhs, Variables<T>& rhs)
	{
		eval_P_x(data, T(1), lhs.x, rhs.x);
		rhs.x.array() += m_rho * lhs.x.array();
		eval_A_xn_and_AT_xt(data, T(1), T(1), lhs.x, lhs.y, rhs.y, work_x);
		rhs.x.array() += work_x.array();
		rhs.y.array() -= m_delta * lhs.y.array();
		rhs.s_l.noalias() = lhs.z_u - lhs.z_l; // use rhs.s_l as temporary
		eval_G_xn_and_GT_xt(data, T(1), T(1), lhs.x, rhs.s_l, rhs.z_u, work_x);
		rhs.z_l.noalias() = -rhs.z_u;
		rhs.x.array() += work_x.array();
		rhs.z_l.array() += lhs.s_l.array() - m_delta * lhs.z_l.array();
		rhs.z_u.array() += lhs.s_u.array() - m_delta * lhs.z_u.array();
		rhs.s_l.array() = m_s_l.array() * lhs.z_l.array() + lhs.s_l.array() / m_z_l_inv.array();
		rhs.s_u.array() = m_s_u.array() * lhs.z_u.array() + lhs.s_u.array() / m_z_u_inv.array();

		for (isize i = 0; i < data.n_x_l; i++)
		{
			Eigen::Index idx = data.x_l_idx(i);
			rhs.x(idx) -= data.x_b_scaling(idx) * lhs.z_bl(i);
			rhs.z_bl(i) = -data.x_b_scaling(idx) * lhs.x(idx) - m_delta * lhs.z_bl(i) + lhs.s_bl(i);
		}
		rhs.s_bl.head(data.n_x_l).array() = m_s_bl.head(data.n_x_l).array() * lhs.z_bl.head(data.n_x_l).array()
			+ lhs.s_bl.head(data.n_x_l).array() / m_z_bl_inv.head(data.n_x_l).array();

		for (isize i = 0; i < data.n_x_u; i++)
		{
			Eigen::Index idx = data.x_u_idx(i);
			rhs.x(idx) += data.x_b_scaling(idx) * lhs.z_bu(i);
			rhs.z_bu(i) = data.x_b_scaling(idx) * lhs.x(idx) - m_delta * lhs.z_bu(i) + lhs.s_bu(i);
		}
		rhs.s_bu.head(data.n_x_u).array() = m_s_bu.head(data.n_x_u).array() * lhs.z_bu.head(data.n_x_u).array()
			+ lhs.s_bu.head(data.n_x_u).array() / m_z_bu_inv.head(data.n_x_u).array();
	}

	void print_info() { kkt_solver->print_info(); }

protected:
	template<int MatrixTypeT = MatrixType>
	std::enable_if_t<MatrixTypeT == PIQP_DENSE>
	extract_P_diag(const DataType& data)
	{
		P_diag.noalias() = data.P_utri.diagonal();
	}

	template<int MatrixTypeT = MatrixType>
	std::enable_if_t<MatrixTypeT == PIQP_SPARSE>
	extract_P_diag(const DataType& data)
	{
		isize jj = data.P_utri.outerSize();
		for (isize j = 0; j < jj; j++)
		{
			isize kk = data.P_utri.outerIndexPtr()[j + 1];
			for (isize k = data.P_utri.outerIndexPtr()[j]; k < kk; k++)
			{
				if (j == data.P_utri.innerIndexPtr()[k])
				{
					P_diag[j] = data.P_utri.valuePtr()[k];
				}
			}
		}
	}

	template<int MatrixTypeT = MatrixType>
	std::enable_if_t<MatrixTypeT == PIQP_DENSE, bool>
	init_kkt_solver(const DataType& data, const Settings<T>& settings)
	{
		switch (settings.kkt_solver) {
			case KKTSolver::dense_cholesky:
				kkt_solver = std::make_unique<dense::KKT<T>>(data);
			break;
			default:
				piqp_eprint("kkt solver not supported\n");
			return false;
		}
		return true;
	}

	template<int MatrixTypeT = MatrixType>
	std::enable_if_t<MatrixTypeT == PIQP_SPARSE, bool>
	init_kkt_solver(const DataType& data, const Settings<T>& settings)
	{
		switch (settings.kkt_solver) {
			case KKTSolver::sparse_ldlt:
				kkt_solver = std::make_unique<sparse::KKT<T, I, KKT_FULL>>(data);
				break;
			case KKTSolver::sparse_ldlt_eq_cond:
				kkt_solver = std::make_unique<sparse::KKT<T, I, KKT_EQ_ELIMINATED>>(data);
				break;
			case KKTSolver::sparse_ldlt_ineq_cond:
				kkt_solver = std::make_unique<sparse::KKT<T, I, KKT_INEQ_ELIMINATED>>(data);
				break;
			case KKTSolver::sparse_ldlt_cond:
				kkt_solver = std::make_unique<sparse::KKT<T, I, KKT_ALL_ELIMINATED>>(data);
				break;
#ifdef PIQP_HAS_BLASFEO
			case KKTSolver::sparse_multistage:
				kkt_solver = std::make_unique<sparse::MultistageKKT<T, I>>(data);
				break;
#endif
			default:
				piqp_eprint("kkt solver not supported\n");
				return false;
		}
		return true;
	}

	T inf_norm(const Vec<T>& x, const Vec<T>& y, const Vec<T>& z)
	{
		T norm = x.template lpNorm<Eigen::Infinity>();
		norm = (std::max)(norm, y.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, z.template lpNorm<Eigen::Infinity>());
		return norm;
	}

	void mul_condensed_kkt(const DataType& data,
                           const Vec<T>& lhs_x, const Vec<T>& lhs_y, const Vec<T>& lhs_z,
						   Vec<T>& rhs_x, Vec<T>& rhs_y, Vec<T>& rhs_z)
	{
		eval_P_x(data, T(1), lhs_x, rhs_x);
		rhs_x.array() += m_x_reg.array() * lhs_x.array();
		eval_A_xn_and_AT_xt(data, T(1), T(1), lhs_x, lhs_y, rhs_y, work_x);
		rhs_x.array() += work_x.array();
		rhs_y.array() -= m_delta * lhs_y.array();
		eval_G_xn_and_GT_xt(data, T(1), T(1), lhs_x, lhs_z, rhs_z, work_x);
		rhs_x.array() += work_x.array();
		rhs_z.array() -= m_z_reg.array() * lhs_z.array();
	}

	// err = rhs - KKT * lhs
	T get_refine_error(const DataType& data,
		              const Vec<T>& lhs_x, const Vec<T>& lhs_y, const Vec<T>& lhs_z,
		              const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z,
		              Vec<T>& err_x, Vec<T>& err_y, Vec<T>& err_z)
	{
		PIQP_TRACY_ZoneScopedN("piqp::KKTSystem::get_refine_error");

		mul_condensed_kkt(data, lhs_x, lhs_y, lhs_z, err_x, err_y, err_z);

		err_x.array() = rhs_x.array() - err_x.array();
		err_y.array() = rhs_y.array() - err_y.array();
		err_z.array() = rhs_z.array() - err_z.array();

		return inf_norm(err_x, err_y, err_z);
	}
};

} // namespace piqp

#endif //PIQP_KKT_SYSTEM_HPP
