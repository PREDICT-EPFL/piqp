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
	Vec<T> err_x;
	Vec<T> err_y;
	Vec<T> err_z;
	Vec<T> err_z_lb;
	Vec<T> err_z_ub;
	Vec<T> err_s;
	Vec<T> err_s_lb;
	Vec<T> err_s_ub;
	Vec<T> ref_delta_x;
	Vec<T> ref_delta_y;
	Vec<T> ref_delta_z;
	Vec<T> ref_delta_z_lb;
	Vec<T> ref_delta_z_ub;
	Vec<T> ref_delta_s;
	Vec<T> ref_delta_s_lb;
	Vec<T> ref_delta_s_ub;

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

		err_x.resize(data.n);
		err_y.resize(data.p);
		err_z.resize(data.m);
		err_z_lb.resize(data.n);
		err_z_ub.resize(data.n);
		err_s.resize(data.m);
		err_s_lb.resize(data.n);
		err_s_ub.resize(data.n);
		ref_delta_x.resize(data.n);
		ref_delta_y.resize(data.p);
		ref_delta_z.resize(data.m);
		ref_delta_z_lb.resize(data.n);
		ref_delta_z_ub.resize(data.n);
		ref_delta_s.resize(data.m);
		ref_delta_s_lb.resize(data.n);
		ref_delta_s_ub.resize(data.n);

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

	bool update_scalings_and_factor(bool iterative_refinement,
									const T& rho, const T& delta,
                                    const Vec<T>& s, const Vec<T>& s_lb, const Vec<T>& s_ub,
                                    const Vec<T>& z, const Vec<T>& z_lb, const Vec<T>& z_ub)
    {
		Vec<T>& m_x_reg = work_x;
		Vec<T>& m_z_reg = work_z;

		m_rho = rho;
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

	bool solve(const Vec<T>& rhs_x, const Vec<T>& rhs_y,
			   const Vec<T>& rhs_z, const Vec<T>& rhs_z_lb, const Vec<T>& rhs_z_ub,
			   const Vec<T>& rhs_s, const Vec<T>& rhs_s_lb, const Vec<T>& rhs_s_ub,
			   Vec<T>& delta_x, Vec<T>& delta_y,
			   Vec<T>& delta_z, Vec<T>& delta_z_lb, Vec<T>& delta_z_ub,
			   Vec<T>& delta_s, Vec<T>& delta_s_lb, Vec<T>& delta_s_ub)
	{
		solve_internal(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub,
					   rhs_s, rhs_s_lb, rhs_s_ub,
			           delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub,
			           delta_s, delta_s_lb, delta_s_ub);

		if (use_iterative_refinement) {

			T rhs_norm = inf_norm(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub, rhs_s, rhs_s_lb, rhs_s_ub);

			T refine_error = get_refine_error(delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub,
											delta_s, delta_s_lb, delta_s_ub,
											rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub,
											rhs_s, rhs_s_lb, rhs_s_ub,
											err_x, err_y, err_z, err_z_lb, err_z_ub,
											err_s, err_s_lb, err_s_ub);

			if (!std::isfinite(refine_error)) return false;

			for (isize i = 0; i < settings.iterative_refinement_max_iter; i++)
			{
				if (refine_error <= settings.iterative_refinement_eps_abs + settings.iterative_refinement_eps_rel * rhs_norm) {
					break;
				}
				T prev_refine_error = refine_error;

				solve_internal(err_x, err_y, err_z, err_z_lb, err_z_ub,
							   err_s, err_s_lb, err_s_ub,
							   ref_delta_x, ref_delta_y, ref_delta_z, ref_delta_z_lb, ref_delta_z_ub,
							   ref_delta_s, ref_delta_s_lb, ref_delta_s_ub);

				// use ref_delta to store refined solution
				ref_delta_x.array() += delta_x.array();
				ref_delta_y.array() += delta_y.array();
				ref_delta_z.array() += delta_z.array();
				ref_delta_z_lb.head(data.n_lb).array() += delta_z_lb.head(data.n_lb).array();
				ref_delta_z_ub.head(data.n_ub).array() += delta_z_ub.head(data.n_ub).array();
				ref_delta_s.array() += delta_s.array();
				ref_delta_s_lb.head(data.n_lb).array() += delta_s_lb.head(data.n_lb).array();
				ref_delta_s_ub.head(data.n_ub).array() += delta_s_ub.head(data.n_ub).array();

				refine_error = get_refine_error(ref_delta_x, ref_delta_y, ref_delta_z, ref_delta_z_lb, ref_delta_z_ub,
											  ref_delta_s, ref_delta_s_lb, ref_delta_s_ub,
											  rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub,
											  rhs_s, rhs_s_lb, rhs_s_ub,
											  err_x, err_y, err_z, err_z_lb, err_z_ub,
											  err_s, err_s_lb, err_s_ub);

				if (!std::isfinite(refine_error)) return false;

				T improvement_rate = prev_refine_error / refine_error;
				if (improvement_rate < settings.iterative_refinement_min_improvement_rate) {
					if (improvement_rate > T(1)) {
						std::swap(delta_x, ref_delta_x);
						std::swap(delta_y, ref_delta_y);
						std::swap(delta_z, ref_delta_z);
						std::swap(delta_z_lb, ref_delta_z_lb);
						std::swap(delta_z_ub, ref_delta_z_ub);
						std::swap(delta_s, ref_delta_s);
						std::swap(delta_s_lb, ref_delta_s_lb);
						std::swap(delta_s_ub, ref_delta_s_ub);
					}
					break;
				}
				std::swap(delta_x, ref_delta_x);
				std::swap(delta_y, ref_delta_y);
				std::swap(delta_z, ref_delta_z);
				std::swap(delta_z_lb, ref_delta_z_lb);
				std::swap(delta_z_ub, ref_delta_z_ub);
				std::swap(delta_s, ref_delta_s);
				std::swap(delta_s_lb, ref_delta_s_lb);
				std::swap(delta_s_ub, ref_delta_s_ub);
			}

		} else {

			if (!delta_x.allFinite() || !delta_y.allFinite() ||
				!delta_z.allFinite() || !delta_z_lb.allFinite() || !delta_z_ub.allFinite() ||
				!delta_s.allFinite() || !delta_s_lb.allFinite() || !delta_s_ub.allFinite()) {
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

	void mul(const Vec<T>& lhs_x, const Vec<T>& lhs_y,
			 const Vec<T>& lhs_z, const Vec<T>& lhs_z_lb, const Vec<T>& lhs_z_ub,
			 const Vec<T>& lhs_s, const Vec<T>& lhs_s_lb, const Vec<T>& lhs_s_ub,
			 Vec<T>& rhs_x, Vec<T>& rhs_y,
			 Vec<T>& rhs_z, Vec<T>& rhs_z_lb, Vec<T>& rhs_z_ub,
			 Vec<T>& rhs_s, Vec<T>& rhs_s_lb, Vec<T>& rhs_s_ub)
	{
		eval_P_x(T(1), lhs_x, rhs_x);
		rhs_x.array() += m_rho;
		eval_A_xn_and_AT_xt(T(1), T(1), lhs_x, lhs_y, rhs_y, work_x);
		rhs_x.array() += work_x.array();
		rhs_y.array() -= m_delta * lhs_y.array();
		eval_G_xn_and_GT_xt(T(1), T(1), lhs_x, lhs_z, rhs_z, work_x);
		rhs_x.array() += work_x.array();
		rhs_z.array() += lhs_s.array() - m_delta * lhs_z.array();
		rhs_s.array() = m_s.array() * lhs_z.array() + lhs_s.array() / m_z_inv.array();

		for (isize i = 0; i < data.n_lb; i++)
		{
			Eigen::Index idx = data.x_lb_idx(i);
			rhs_x(idx) -= data.x_b_scaling(idx) * lhs_z_lb(i);
			rhs_z_lb(i) = -data.x_b_scaling(idx) * lhs_x(idx) - m_delta * lhs_z_lb(i) + lhs_s_lb(i);
		}
		rhs_s_lb.head(data.n_lb).array() = m_s_lb.head(data.n_lb).array() * lhs_z_lb.head(data.n_lb).array()
			+ lhs_s_lb.head(data.n_lb).array() / m_z_lb_inv.head(data.n_lb).array();

		for (isize i = 0; i < data.n_ub; i++)
		{
			Eigen::Index idx = data.x_ub_idx(i);
			rhs_x(idx) += data.x_b_scaling(idx) * lhs_z_ub(i);
			rhs_z_ub(i) = data.x_b_scaling(idx) * lhs_x(idx) - m_delta * lhs_z_ub(i) + lhs_s_ub(i);
		}
		rhs_s_ub.head(data.n_ub).array() = m_s_ub.head(data.n_ub).array() * lhs_z_ub.head(data.n_ub).array()
			+ lhs_s_ub.head(data.n_ub).array() / m_z_ub_inv.head(data.n_ub).array();
	}

	void print_info() { kkt_solver->print_info(); }

protected:
	template<int MatrixTypeT = MatrixType>
	typename std::enable_if<MatrixTypeT == PIQP_DENSE>::type
	extract_P_diag()
	{
		P_diag.noalias() = data.P_utri.diagonal();
	}

	template<int MatrixTypeT = MatrixType>
	typename std::enable_if<MatrixTypeT == PIQP_SPARSE>::type
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

	void solve_internal(const Vec<T>& rhs_x, const Vec<T>& rhs_y,
						const Vec<T>& rhs_z, const Vec<T>& rhs_z_lb, const Vec<T>& rhs_z_ub,
					    const Vec<T>& rhs_s, const Vec<T>& rhs_s_lb, const Vec<T>& rhs_s_ub,
					    Vec<T>& delta_x, Vec<T>& delta_y,
					    Vec<T>& delta_z, Vec<T>& delta_z_lb, Vec<T>& delta_z_ub,
					    Vec<T>& delta_s, Vec<T>& delta_s_lb, Vec<T>& delta_s_ub)
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

	T inf_norm(const Vec<T>& x, const Vec<T>& y,
			   const Vec<T>& z, const Vec<T>& z_lb, const Vec<T>& z_ub,
			   const Vec<T>& s, const Vec<T>& s_lb, const Vec<T>& s_ub)
	{
		T norm = x.template lpNorm<Eigen::Infinity>();
		norm = (std::max)(norm, y.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, z.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, z_lb.head(data.n_lb).template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, z_ub.head(data.n_ub).template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, s.template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, s_lb.head(data.n_lb).template lpNorm<Eigen::Infinity>());
		norm = (std::max)(norm, s_ub.head(data.n_ub).template lpNorm<Eigen::Infinity>());
		return norm;
	}

	// err = rhs - KKT * lhs
	T get_refine_error(const Vec<T>& lhs_x, const Vec<T>& lhs_y,
					  const Vec<T>& lhs_z, const Vec<T>& lhs_z_lb, const Vec<T>& lhs_z_ub,
					  const Vec<T>& lhs_s, const Vec<T>& lhs_s_lb, const Vec<T>& lhs_s_ub,
					  const Vec<T>& rhs_x, const Vec<T>& rhs_y,
					  const Vec<T>& rhs_z, const Vec<T>& rhs_z_lb, const Vec<T>& rhs_z_ub,
					  const Vec<T>& rhs_s, const Vec<T>& rhs_s_lb, const Vec<T>& rhs_s_ub,
					  Vec<T>& err_x, Vec<T>& err_y,
					  Vec<T>& err_z, Vec<T>& err_z_lb, Vec<T>& err_z_ub,
					  Vec<T>& err_s, Vec<T>& err_s_lb, Vec<T>& err_s_ub)
	{
		mul(lhs_x, lhs_y, lhs_z, lhs_z_lb, lhs_z_ub,
			lhs_s, lhs_s_lb, lhs_s_ub,
			err_x, err_y, err_z, err_z_lb, err_z_ub,
			err_s, err_s_lb, err_s_ub);
		err_x.array() = rhs_x.array() - err_x.array();
		err_y.array() = rhs_y.array() - err_y.array();
		err_z.array() = rhs_z.array() - err_z.array();
		err_z_lb.head(data.n_lb).array() = rhs_z_lb.head(data.n_lb).array() - err_z_lb.head(data.n_lb).array();
		err_z_ub.head(data.n_ub).array() = rhs_z_ub.head(data.n_ub).array() - err_z_ub.head(data.n_ub).array();
		err_s.array() = rhs_s.array() - err_s.array();
		err_s_lb.head(data.n_lb).array() = rhs_s_lb.head(data.n_lb).array() - err_s_lb.head(data.n_lb).array();
		err_s_ub.head(data.n_ub).array() = rhs_s_ub.head(data.n_ub).array() - err_s_ub.head(data.n_ub).array();

		return inf_norm(err_x, err_y, err_z, err_z_lb, err_z_ub,
						err_s, err_s_lb, err_s_ub);
	}
};

} // namespace piqp

#endif //PIQP_KKT_SYSTEM_HPP
