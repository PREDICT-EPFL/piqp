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

namespace piqp
{

template<typename T, typename I, int MatrixType, int Mode = KKT_FULL>
class KKTSystem
{
protected:
	using DataType = std::conditional_t<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>;

	const DataType& data;
	const Settings<T>& settings;

	T m_rho;
	T m_delta;

	Vec<T> P_diag;

	Vec<T> m_s;
	Vec<T> m_s_lb;
	Vec<T> m_s_ub;
	Vec<T> m_z_inv;
	Vec<T> m_z_lb_inv;
	Vec<T> m_z_ub_inv;

	// working variables
	Vec<T> work_x;
	Vec<T> work_z;

	// iterative refinement variables
	Variables<T> ref_err;
	Variables<T> ref_lhs;

	bool use_iterative_refinement;
	std::unique_ptr<KKTSolverBase<T>> kkt_solver;

public:
	KKTSystem(const DataType& data, const Settings<T>& settings) : data(data), settings(settings),
	                                                               use_iterative_refinement(false) {}

	bool init()
	{
		P_diag.resize(data.n);
		P_diag.setZero();

		m_s.resize(data.m);
		m_s_lb.resize(data.n);
		m_s_ub.resize(data.n);
		m_z_inv.resize(data.m);
		m_z_lb_inv.resize(data.n);
		m_z_ub_inv.resize(data.n);

		work_x.resize(data.n);
		work_z.resize(data.m);

		ref_err.resize(data.n, data.p, data.m);
		ref_lhs.resize(data.n, data.p, data.m);

		extract_P_diag();

		return init_kkt_solver<MatrixType>();
	}

	void update_data(int options)
	{
		if (options & KKTUpdateOptions::KKT_UPDATE_P) {
			extract_P_diag();
		}

		kkt_solver->update_data(options);
	}

	bool update_scalings_and_factor(bool iterative_refinement, const T& rho, const T& delta, const Variables<T>& vars)
    {
		Vec<T>& m_x_reg = work_x;
		Vec<T>& m_z_reg = work_z;

		m_rho = rho;
		m_delta = delta;
		m_s = vars.s;
		m_s_lb.head(data.n_lb) = vars.s_lb.head(data.n_lb);
		m_s_ub.head(data.n_ub) = vars.s_ub.head(data.n_ub);
		m_z_inv.array() = vars.z.array().inverse();
		m_z_lb_inv.head(data.n_lb).array() = vars.z_lb.head(data.n_lb).array().inverse();
		m_z_ub_inv.head(data.n_ub).array() = vars.z_ub.head(data.n_ub).array().inverse();

		m_x_reg.setConstant(rho);
		for (isize i = 0; i < data.n_lb; i++)
		{
			Eigen::Index idx = data.x_lb_idx(i);
			m_x_reg(idx) += data.x_b_scaling(idx) * data.x_b_scaling(idx) / (m_z_lb_inv(i) * m_s_lb(i) + m_delta);
		}
		for (isize i = 0; i < data.n_ub; i++)
		{
			Eigen::Index idx = data.x_ub_idx(i);
			m_x_reg(idx) += data.x_b_scaling(idx) * data.x_b_scaling(idx) / (m_z_ub_inv(i) * m_s_ub(i) + m_delta);
		}

		m_z_reg.array() = m_s.array() * m_z_inv.array() + delta;

		if (iterative_refinement)
		{
			T max_diag = (P_diag + m_x_reg).template lpNorm<Eigen::Infinity>();
			max_diag = (std::max)(max_diag, m_z_reg.template lpNorm<Eigen::Infinity>());

			T reg = settings.iterative_refinement_static_regularization_eps
				    + settings.iterative_refinement_static_regularization_rel * max_diag;

			m_x_reg.array() += reg;
			m_z_reg.array() += reg;
		}

		use_iterative_refinement = iterative_refinement;
		return kkt_solver->update_scalings_and_factor(delta, m_x_reg, m_z_reg);
    }

	bool solve(const Variables<T>& rhs, Variables<T>& lhs)
	{
		solve_internal(rhs, lhs);

		if (use_iterative_refinement) {

			T rhs_norm = inf_norm(rhs);

			T refine_error = get_refine_error(lhs, rhs, ref_err);

			if (!std::isfinite(refine_error)) return false;

			for (isize i = 0; i < settings.iterative_refinement_max_iter; i++)
			{
				if (refine_error <= settings.iterative_refinement_eps_abs + settings.iterative_refinement_eps_rel * rhs_norm) {
					break;
				}
				T prev_refine_error = refine_error;

				solve_internal(ref_err, ref_lhs);

				// use ref_lhs to store refined solution
				ref_lhs += lhs;

				refine_error = get_refine_error(ref_lhs, rhs, ref_err);

				if (!std::isfinite(refine_error)) return false;

				T improvement_rate = prev_refine_error / refine_error;
				if (improvement_rate < settings.iterative_refinement_min_improvement_rate) {
					if (improvement_rate > T(1)) {
						std::swap(lhs, ref_lhs);
					}
					break;
				}
				std::swap(lhs, ref_lhs);
			}

		} else {

			if (!lhs.allFinite()) {
				return false;
			}
		}

		return true;
	}

	// z = alpha * P * x
	void eval_P_x(const T& alpha, const Vec<T>& x, Vec<T>& z)
	{
		kkt_solver->eval_P_x(alpha, x, z);
	}

	// zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
	void eval_A_xn_and_AT_xt(const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
	{
		kkt_solver->eval_A_xn_and_AT_xt(alpha_n, alpha_t, xn, xt, zn, zt);
	}

	// zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
	void eval_G_xn_and_GT_xt(const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
	{
		kkt_solver->eval_G_xn_and_GT_xt(alpha_n, alpha_t, xn, xt, zn, zt);
	}

	void mul(const Variables<T>& lhs, Variables<T>& rhs)
	{
		eval_P_x(T(1), lhs.x, rhs.x);
		rhs.x.array() += m_rho * lhs.x.array();
		eval_A_xn_and_AT_xt(T(1), T(1), lhs.x, lhs.y, rhs.y, work_x);
		rhs.x.array() += work_x.array();
		rhs.y.array() -= m_delta * lhs.y.array();
		eval_G_xn_and_GT_xt(T(1), T(1), lhs.x, lhs.z, rhs.z, work_x);
		rhs.x.array() += work_x.array();
		rhs.z.array() += lhs.s.array() - m_delta * lhs.z.array();
		rhs.s.array() = m_s.array() * lhs.z.array() + lhs.s.array() / m_z_inv.array();

		for (isize i = 0; i < data.n_lb; i++)
		{
			Eigen::Index idx = data.x_lb_idx(i);
			rhs.x(idx) -= data.x_b_scaling(idx) * lhs.z_lb(i);
			rhs.z_lb(i) = -data.x_b_scaling(idx) * lhs.x(idx) - m_delta * lhs.z_lb(i) + lhs.s_lb(i);
		}
		rhs.s_lb.head(data.n_lb).array() = m_s_lb.head(data.n_lb).array() * lhs.z_lb.head(data.n_lb).array()
			+ lhs.s_lb.head(data.n_lb).array() / m_z_lb_inv.head(data.n_lb).array();

		for (isize i = 0; i < data.n_ub; i++)
		{
			Eigen::Index idx = data.x_ub_idx(i);
			rhs.x(idx) += data.x_b_scaling(idx) * lhs.z_ub(i);
			rhs.z_ub(i) = data.x_b_scaling(idx) * lhs.x(idx) - m_delta * lhs.z_ub(i) + lhs.s_ub(i);
		}
		rhs.s_ub.head(data.n_ub).array() = m_s_ub.head(data.n_ub).array() * lhs.z_ub.head(data.n_ub).array()
			+ lhs.s_ub.head(data.n_ub).array() / m_z_ub_inv.head(data.n_ub).array();
	}

	void print_info() { kkt_solver->print_info(); }

protected:
	template<int MatrixTypeT = MatrixType>
	std::enable_if_t<MatrixTypeT == PIQP_DENSE>
	extract_P_diag()
	{
		P_diag.noalias() = data.P_utri.diagonal();
	}

	template<int MatrixTypeT = MatrixType>
	std::enable_if_t<MatrixTypeT == PIQP_SPARSE>
	extract_P_diag()
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
	std::enable_if_t<MatrixTypeT == PIQP_SPARSE, bool>
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

	void solve_internal(const Variables<T>& rhs, Variables<T>& lhs)
	{
		Vec<T>& rhs_x_bar = work_x;
		Vec<T>& rhs_z_bar = work_z;

		rhs_z_bar.array() = rhs.z.array() - m_z_inv.array() * rhs.s.array();

		rhs_x_bar = rhs.x;
		for (isize i = 0; i < data.n_lb; i++)
		{
			Eigen::Index idx = data.x_lb_idx(i);
			rhs_x_bar(idx) -= data.x_b_scaling(idx) * (rhs.z_lb(i) - m_z_lb_inv(i) * rhs.s_lb(i))
				/ (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
		}
		for (isize i = 0; i < data.n_ub; i++)
		{
			Eigen::Index idx = data.x_ub_idx(i);
			rhs_x_bar(idx) += data.x_b_scaling(idx) * (rhs.z_ub(i) - m_z_ub_inv(i) * rhs.s_ub(i))
				/ (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
		}

		kkt_solver->solve(rhs_x_bar, rhs.y, rhs_z_bar, lhs.x, lhs.y, lhs.z);

		for (isize i = 0; i < data.n_lb; i++) {
			Eigen::Index idx = data.x_lb_idx(i);
			lhs.z_lb(i) = (-data.x_b_scaling(idx) * lhs.x(idx) - rhs.z_lb(i) + m_z_lb_inv(i) * rhs.s_lb(i))
				/ (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
		}
		for (isize i = 0; i < data.n_ub; i++) {
			Eigen::Index idx = data.x_ub_idx(i);
			lhs.z_ub(i) = (data.x_b_scaling(idx) * lhs.x(idx) - rhs.z_ub(i) + m_z_ub_inv(i) * rhs.s_ub(i))
				/ (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
		}

		lhs.s.array() = m_z_inv.array() * (rhs.s.array() - m_s.array() * lhs.z.array());

		lhs.s_lb.head(data.n_lb).array() = m_z_lb_inv.head(data.n_lb).array()
			* (rhs.s_lb.head(data.n_lb).array() - m_s_lb.head(data.n_lb).array() * lhs.z_lb.head(data.n_lb).array());

		lhs.s_ub.head(data.n_ub).array() = m_z_ub_inv.head(data.n_ub).array()
			* (rhs.s_ub.head(data.n_ub).array() - m_s_ub.head(data.n_ub).array() * lhs.z_ub.head(data.n_ub).array());
	}

	T inf_norm(const Variables<T>& vars)
	{
		T norm = vars.x.template lpNorm<Eigen::Infinity>();
		norm = (std::max)(norm, vars.y.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, vars.z.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, vars.z_lb.head(data.n_lb).template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, vars.z_ub.head(data.n_ub).template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, vars.s.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, vars.s_lb.head(data.n_lb).template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, vars.s_ub.head(data.n_ub).template lpNorm<Eigen::Infinity>());
		return norm;
	}

	// err = rhs - KKT * lhs
	T get_refine_error(const Variables<T>& lhs, const Variables<T>& rhs, Variables<T>& err)
	{
		mul(lhs, err);
		err.x.array() = rhs.x.array() - err.x.array();
		err.y.array() = rhs.y.array() - err.y.array();
		err.z.array() = rhs.z.array() - err.z.array();
		err.z_lb.head(data.n_lb).array() = rhs.z_lb.head(data.n_lb).array() - err.z_lb.head(data.n_lb).array();
		err.z_ub.head(data.n_ub).array() = rhs.z_ub.head(data.n_ub).array() - err.z_ub.head(data.n_ub).array();
		err.s.array() = rhs.s.array() - err.s.array();
		err.s_lb.head(data.n_lb).array() = rhs.s_lb.head(data.n_lb).array() - err.s_lb.head(data.n_lb).array();
		err.s_ub.head(data.n_ub).array() = rhs.s_ub.head(data.n_ub).array() - err.s_ub.head(data.n_ub).array();

		return inf_norm(err);
	}
};

} // namespace piqp

#endif //PIQP_KKT_SYSTEM_HPP
