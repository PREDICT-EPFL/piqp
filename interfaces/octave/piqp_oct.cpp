#include <octave/oct.h>
#include "piqp.hpp"

// LD_PRELOAD=/home/redstone/tmp/piqp/interfaces/c/libpiqpc.so octave
// autoload("piqp_dense", "/home/redstone/tmp/piqp/interfaces/octave/piqp.oct")
// piqp_dense
static_assert(sizeof(piqp_float) == 8);

DEFUN_DLD (piqp_dense, args, nargout,
           "piqp_dense(x0, Q, c)")
{
  octave_stdout << "piqp_dense has "
                << args.length () << " input arguments and "
                << nargout << " output arguments.\n";
  // Return empty matrices for any outputs
  octave_value_list retval (nargout);
  for (int i = 0; i < nargout; i++)
    retval(i) = octave_value (Matrix ());

  if (args.length() != 3) {
    error("piqp_dense: Incorrect # of args#");
    return retval;
  }
 
  piqp_workspace* work;

  piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
  piqp_set_default_settings(settings);
  settings->verbose = 1;

  ColumnVector x0  (args(0).vector_value ());
  Matrix        Q  (args(1).matrix_value ());
  ColumnVector  c  (args(2).vector_value ());

  int n = x0.numel();
  // For Matrix, data() goes by columns.
  printf("Qa is #%ld  and %g %g %g\n", Q.numel(), Q.data()[0], Q.data()[1], Q.data()[2]);
  printf("Q is #%ld  and %g %g %g\n", Q.numel(), Q.fortran_vec()[0], Q.fortran_vec()[1], Q.fortran_vec()[2]);
  return retval;

  
  piqp_data_dense* data = (piqp_data_dense*) malloc(sizeof(piqp_data_dense));
  data->n = n;

  data->p = 0;  // # equality constraints
  data->m = 0; // # inequality constraints
  //data->P = Q;
  data->c = c.fortran_vec(); // linear cost weights

#if 0
  data->A = A;
  data->b = b;
  data->G = G;
  data->h = h;
  data->x_lb = x_lb;
  data->x_ub = x_ub;
#endif    
  piqp_setup_dense(&work, data, settings);
  piqp_status status = piqp_solve(work);

  printf("status = %d\n", status);
  printf("x = %f %f\n", work->result->x[0], work->result->x[1]);

  piqp_cleanup(work);
  free(settings);
  free(data);
 

  return retval;
}
