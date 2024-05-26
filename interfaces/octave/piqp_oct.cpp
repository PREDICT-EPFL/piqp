#include <limits>
#include <octave/oct.h>
#include "ovl.h" // for octave value list
#include "piqp.hpp"

// cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_OCTAVE_INTERFACE=ON .
// make piqp.oct
//
// LD_PRELOAD=/home/redstone/tmp/piqp/interfaces/c/libpiqpc.so octave
// autoload("piqp_dense", "/home/redstone/tmp/piqp/interfaces/octave/piqp.oct")
// [res] = piqp_dense([1 0 ; 0 1], [-4 ; -6], [], [], [], [], [-Inf ; -Inf], [Inf; Inf], struct("verbose", true))

//static_assert(sizeof(piqp_float) == 8);

ColumnVector eigen3VecToCol(const Eigen::Matrix<double, Eigen::Dynamic, 1>& vec) {
  int n = vec.size();
  double* vraw = new double[n];
  memcpy(vraw, vec.data(), sizeof(double)*n);
  ColumnVector x(Array<double>(vraw, dim_vector(n,1)));
  return x;
}

DEFUN_DLD (piqp_dense, args, nargout,
           "piqp_dense(Q, c, A, b, G, h, x_lb, x_ub, opts)")
{
  octave_stdout << "piqp_dense has "
                << args.length () << " input arguments and "
                << nargout << " output arguments.\n";
  // Return empty matrices for any outputs
  octave_value_list retval (nargout);
  for (int i = 0; i < nargout; i++)
    retval(i) = octave_value (Matrix ());

  if (args.length() < 8) {
    error("piqp_dense: Incorrect # of args#");
    return retval;
  }
 
  int n = args(1).vector_value().numel();
  int neq = args(3).vector_value().numel();
  int nineq = args(5).vector_value().numel();

  Eigen::MatrixXd Q = Eigen::Map<Eigen::MatrixXd>(args(0).matrix_value().fortran_vec(), n, n);
  printf("happy Q(0,1)=%g  Q(1,0)=%g vs %g %d %d \n", Q(0,1), Q(1,0), std::numeric_limits<double>::infinity(),
         octave::math::isinf(Q(0,1)), octave::math::isinf(std::numeric_limits<double>::infinity()));
  
  Eigen::VectorXd c = Eigen::Map<Eigen::VectorXd>(args(1).vector_value().fortran_vec(), n, 1);
  if (0) {
    double* craw = new double[n];
    memcpy(craw, c.data(), sizeof(double)*n);
    ColumnVector cc(Array<double>(craw, dim_vector(n,1)));
    printf("c has %g %g\n", cc(0), cc(1));
    return retval;
  }

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
  
  //std::cout << "status = " << status << std::endl;
  //std::cout << "x = " << solver.result().x.transpose() << std::endl;

  //double* xraw = new double[n];
  //memcpy(xraw, solver.result().x.data(), sizeof(double)*n);
  //ColumnVector x(Array<double>(xraw, dim_vector(n,1)));

  int numiter = solver.result().info.iter;
  double obj = solver.result().info.primal_obj;
  octave_scalar_map info;
#define iset(a) info.assign(#a, solver.result().info.a);
  iset(status);
  iset(iter);
  iset(rho);
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
