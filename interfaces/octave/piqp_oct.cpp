// This file is part of PIQP.
//
// Copyright (c) 2024 Joshua Redstone
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <limits>
#include <octave/oct.h>
#include "ovl.h" // for octave value list
#include "piqp.hpp"

ColumnVector eigen3VecToCol(const Eigen::Matrix<double, Eigen::Dynamic, 1>& vec) {
  int n = vec.size();
  double* vraw = new double[n];
  memcpy(vraw, vec.data(), sizeof(double)*n);
  ColumnVector x(Array<double>(vraw, dim_vector(n,1)));
  return x;
}

DEFUN_DLD (__piqp__, args, nargout,
           "rez = __piqp__(Q, c, A, b, G, h, x_lb, x_ub, opts)\nOnly supports dense matrices at present")
{
  if (args.length() < 8) {
    error("piqp_dense: Incorrect # of args#");
    return ovl();
  }
 
  int n = args(1).vector_value().numel();
  int neq = args(3).vector_value().numel();
  int nineq = args(5).vector_value().numel();

  // Convert octave matrix and vectors to Eigen versions of the same
  Eigen::MatrixXd Q = Eigen::Map<Eigen::MatrixXd>(args(0).matrix_value().fortran_vec(), n, n);
  Eigen::VectorXd c = Eigen::Map<Eigen::VectorXd>(args(1).vector_value().fortran_vec(), n, 1);
  Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(args(2).matrix_value().fortran_vec(), neq, n);
  Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(args(3).vector_value().fortran_vec(), neq, 1);
  Eigen::MatrixXd G = Eigen::Map<Eigen::MatrixXd>(args(4).matrix_value().fortran_vec(), nineq, n);
  Eigen::VectorXd h = Eigen::Map<Eigen::VectorXd>(args(5).vector_value().fortran_vec(), nineq, 1);
  Eigen::VectorXd x_lb = Eigen::Map<Eigen::VectorXd>(args(6).vector_value().fortran_vec(), n, 1);
  Eigen::VectorXd x_ub = Eigen::Map<Eigen::VectorXd>(args(7).vector_value().fortran_vec(), n, 1);

  piqp::DenseSolver<double> solver;

  if (args.length() == 9) {
    const octave_scalar_map& opts = args(8).scalar_map_value();
#define DFIELD(a) { if (opts.contains(#a)) {                            \
        solver.settings().a = opts.getfield(#a).double_value(); } }
#define BFIELD(a) { if (opts.contains(#a)) {                            \
        solver.settings().a = opts.getfield(#a).bool_value(); } }
#define IFIELD(a) { if (opts.contains(#a)) {                            \
        solver.settings().a = opts.getfield(#a).int_value(); } }
    DFIELD(rho_init);
    DFIELD(delta_init);
    DFIELD(eps_abs);
    DFIELD(eps_rel);
    BFIELD(check_duality_gap);
    DFIELD(eps_duality_gap_abs);
    DFIELD(eps_duality_gap_rel);
    DFIELD(reg_lower_limit);
    DFIELD(reg_finetune_lower_limit);
    IFIELD(reg_finetune_primal_update_threshold);
    IFIELD(reg_finetune_dual_update_threshold);
    IFIELD(max_iter);
    IFIELD(max_factor_retires);
    BFIELD(preconditioner_scale_cost);
    IFIELD(preconditioner_iter);
    DFIELD(tau);

    BFIELD(iterative_refinement_always_enabled);
    DFIELD(iterative_refinement_eps_abs);
    DFIELD(iterative_refinement_eps_rel);
    IFIELD(iterative_refinement_max_iter);
    DFIELD(iterative_refinement_min_improvement_rate);
    DFIELD(iterative_refinement_static_regularization_eps);
    DFIELD(iterative_refinement_static_regularization_rel);

    BFIELD(verbose);
    BFIELD(compute_timings);
#undef BFIELD
#undef DFIELD
#undef IFIELD
  }

  solver.setup(Q, c, A, b, G, h, x_lb, x_ub);
  piqp::Status status = solver.solve();
  
  octave_scalar_map info;
#define iset(a) info.assign(#a, solver.result().info.a);
  // Seems like if first element assigned to map is numeric, it's prints better for some reason.
  // Or maybe it's the info.status Status enum that confused things.
  iset(rho);
  info.assign("status", static_cast<int>(solver.result().info.status));
  iset(iter);
  iset(delta);
  iset(mu);
  iset(sigma);
  iset(primal_step);
  iset(dual_step);
  iset(primal_inf);
  iset(primal_rel_inf);
  iset(dual_inf);
  iset(dual_rel_inf);
  iset(primal_obj);
  iset(dual_obj);
  iset(duality_gap);
  iset(duality_gap_rel);
  iset(factor_retires);
  iset(reg_limit);
  iset(no_primal_update);
  iset(no_dual_update);
  iset(setup_time);
  iset(update_time);
  iset(solve_time);
  iset(run_time);
#undef iset
  octave_scalar_map res;
  res.assign("info", info);
#define rset(a) res.assign(#a, eigen3VecToCol(solver.result().a));
  rset(x);
  rset(y);
  rset(z);
  rset(z_lb);
  rset(z_ub);
  rset(s);
  rset(s_lb);
  rset(s_ub);
  rset(zeta);
  rset(lambda);
  rset(nu);
  rset(nu_lb);
  rset(nu_ub);
#undef rset
  return ovl(res);
}
