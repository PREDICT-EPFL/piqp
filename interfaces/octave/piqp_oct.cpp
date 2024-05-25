#include <octave/oct.h>
#include "piqp.hpp"

// cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_OCTAVE_INTERFACE=ON .
// make piqp.oct
//
// LD_PRELOAD=/home/redstone/tmp/piqp/interfaces/c/libpiqpc.so octave
// autoload("piqp_dense", "/home/redstone/tmp/piqp/interfaces/octave/piqp.oct")
// piqp_dense

//static_assert(sizeof(piqp_float) == 8);

DEFUN_DLD (piqp_dense, args, nargout,
           "piqp_dense(Q, c, A, b, G, h)")
{
  octave_stdout << "piqp_dense has "
                << args.length () << " input arguments and "
                << nargout << " output arguments.\n";
  // Return empty matrices for any outputs
  octave_value_list retval (nargout);
  for (int i = 0; i < nargout; i++)
    retval(i) = octave_value (Matrix ());

  if (args.length() != 6) {
    error("piqp_dense: Incorrect # of args#");
    return retval;
  }
 
  int n = args(1).vector_value().numel();
  int neq = args(3).vector_value().numel();
  int nineq = args(5).vector_value().numel();

  Eigen::MatrixXd Q = Eigen::Map<Eigen::MatrixXd>(args(0).matrix_value().fortran_vec(), n, n);
  Eigen::VectorXd c = Eigen::Map<Eigen::VectorXd>(args(1).vector_value().fortran_vec(), n, 1);
  Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(args(2).matrix_value().fortran_vec(), neq, n);
  Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(args(3).vector_value().fortran_vec(), neq, 1);
  Eigen::MatrixXd G = Eigen::Map<Eigen::MatrixXd>(args(4).matrix_value().fortran_vec(), nineq, n);
  Eigen::VectorXd h = Eigen::Map<Eigen::VectorXd>(args(5).vector_value().fortran_vec(), nineq, 1);

  printf("happy Q(0,1)=%g  Q(1,0)=%g vs %g\n", Q(0,1), Q(1,0), std::numeric_limits<double>::infinity());
  
  //ColumnVector x0  (args(0).vector_value ());
  //Matrix        Q  (args(1).matrix_value ());
  //ColumnVector  c  (args(2).vector_value ());

  //  int n = x0.numel();
  // For Matrix, data() goes by columns.
  //printf("Qa is #%ld  and %g %g %g\n", Q.numel(), Q.data()[0], Q.data()[1], Q.data()[2]);
  //printf("Q is #%ld  and %g %g %g\n", Q.numel(), Q.fortran_vec()[0], Q.fortran_vec()[1], Q.fortran_vec()[2]);
  return retval;

  
}
