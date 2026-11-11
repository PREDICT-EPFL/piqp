// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "piqp/piqp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

#ifndef PIQP_STD_OPTIONAL
template<typename T>
struct py::detail::type_caster<tl::optional<T>> : public py::detail::optional_caster<tl::optional<T>> {};
template<>
struct py::detail::type_caster<tl::nullopt_t> : public py::detail::void_caster<tl::nullopt_t> {};
#endif

using T = double;
using I = int;

// scipy.sparse.csc_matrix returned by sparse_matrix_view(), the class gives the correct type in the signature
template<typename Scalar>
class ScipyCscMatrix : public py::object
{
    static bool check(PyObject*) { return true; }
public:
    PYBIND11_OBJECT_DEFAULT(ScipyCscMatrix, object, check)
};

// same type name as pybind11's Eigen sparse caster, e.g. scipy.sparse.csc_matrix[numpy.float64]
template<typename Scalar>
struct py::detail::handle_type_name<ScipyCscMatrix<Scalar>>
{
    static constexpr auto name = py::detail::const_name("scipy.sparse.csc_matrix[")
                                 + py::detail::npy_format_descriptor<Scalar>::name
                                 + py::detail::const_name("]");
};

// Read-only scipy.sparse.csc_matrix view into the compressed sparse matrix mat, which is owned by base.
// The view stays valid as long as the storage of mat is not reallocated.
template<typename SparseMatType>
ScipyCscMatrix<typename SparseMatType::Scalar> sparse_matrix_view(const SparseMatType& mat, py::handle base)
{
    using Scalar = typename SparseMatType::Scalar;
    using StorageIndex = typename SparseMatType::StorageIndex;
    static_assert(!SparseMatType::IsRowMajor, "sparse_matrix_view requires a column-major matrix");
    if (!mat.isCompressed()) throw std::runtime_error("sparse_matrix_view requires a compressed matrix");

    py::array data = py::array_t<Scalar>(mat.nonZeros(), mat.valuePtr(), base);
    py::array indices = py::array_t<StorageIndex>(mat.nonZeros(), mat.innerIndexPtr(), base);
    py::array indptr = py::array_t<StorageIndex>(mat.outerSize() + 1, mat.outerIndexPtr(), base);
    
    data.attr("flags").attr("writeable") = false;
    indices.attr("flags").attr("writeable") = false;
    indptr.attr("flags").attr("writeable") = false;

    py::object csc = py::module_::import("scipy.sparse").attr("csc_matrix")(
        py::make_tuple(data, indices, indptr),
        py::arg("shape") = py::make_tuple(mat.rows(), mat.cols()),
        py::arg("copy") = false);
    return csc.cast<ScipyCscMatrix<Scalar>>();
}

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
    py::enum_<piqp::Status>(m, "Status", py::module_local())
        .value("PIQP_SOLVED", piqp::Status::PIQP_SOLVED)
        .value("PIQP_MAX_ITER_REACHED", piqp::Status::PIQP_MAX_ITER_REACHED)
        .value("PIQP_PRIMAL_INFEASIBLE", piqp::Status::PIQP_PRIMAL_INFEASIBLE)
        .value("PIQP_DUAL_INFEASIBLE", piqp::Status::PIQP_DUAL_INFEASIBLE)
        .value("PIQP_NUMERICS", piqp::Status::PIQP_NUMERICS)
        .value("PIQP_UNSOLVED", piqp::Status::PIQP_UNSOLVED)
        .value("PIQP_INVALID_SETTINGS", piqp::Status::PIQP_INVALID_SETTINGS)
        .export_values();

    py::class_<piqp::Info<T>>(m, "Info", py::module_local())
        .def(py::init<>())
        .def_readwrite("status", &piqp::Info<T>::status)
        .def_readwrite("iter", &piqp::Info<T>::iter)
        .def_readwrite("rho", &piqp::Info<T>::rho)
        .def_readwrite("delta", &piqp::Info<T>::delta)
        .def_readwrite("mu", &piqp::Info<T>::mu)
        .def_readwrite("sigma", &piqp::Info<T>::sigma)
        .def_readwrite("primal_step", &piqp::Info<T>::primal_step)
        .def_readwrite("dual_step", &piqp::Info<T>::dual_step)
        .def_readwrite("primal_res", &piqp::Info<T>::primal_res)
        .def_readwrite("primal_res_rel", &piqp::Info<T>::primal_res_rel)
        .def_readwrite("dual_res", &piqp::Info<T>::dual_res)
        .def_readwrite("dual_res_rel", &piqp::Info<T>::dual_res_rel)
        .def_readwrite("primal_res_reg", &piqp::Info<T>::primal_res_reg)
        .def_readwrite("primal_res_reg_rel", &piqp::Info<T>::primal_res_reg_rel)
        .def_readwrite("dual_res_reg", &piqp::Info<T>::dual_res_reg)
        .def_readwrite("dual_res_reg_rel", &piqp::Info<T>::dual_res_reg_rel)
        .def_readwrite("primal_prox_inf", &piqp::Info<T>::primal_prox_inf)
        .def_readwrite("dual_prox_inf", &piqp::Info<T>::dual_prox_inf)
        .def_readwrite("prev_primal_res", &piqp::Info<T>::prev_primal_res)
        .def_readwrite("prev_dual_res", &piqp::Info<T>::prev_dual_res)
        .def_readwrite("primal_obj", &piqp::Info<T>::primal_obj)
        .def_readwrite("dual_obj", &piqp::Info<T>::dual_obj)
        .def_readwrite("duality_gap", &piqp::Info<T>::duality_gap)
        .def_readwrite("duality_gap_rel", &piqp::Info<T>::duality_gap_rel)
        .def_readwrite("factor_retires", &piqp::Info<T>::factor_retires)
        .def_readwrite("reg_limit", &piqp::Info<T>::reg_limit)
        .def_readwrite("no_primal_update", &piqp::Info<T>::no_primal_update)
        .def_readwrite("no_dual_update", &piqp::Info<T>::no_dual_update)
        .def_readwrite("setup_time", &piqp::Info<T>::setup_time)
        .def_readwrite("update_time", &piqp::Info<T>::update_time)
        .def_readwrite("solve_time", &piqp::Info<T>::solve_time)
        .def_readwrite("kkt_factor_time", &piqp::Info<T>::kkt_factor_time)
        .def_readwrite("kkt_solve_time", &piqp::Info<T>::kkt_solve_time)
        .def_readwrite("run_time", &piqp::Info<T>::run_time);

    py::class_<piqp::Result<T>>(m, "Result", py::module_local())
        .def(py::init<>())
        .def_readwrite("x", &piqp::Result<T>::x)
        .def_readwrite("y", &piqp::Result<T>::y)
        .def_readwrite("z_l", &piqp::Result<T>::z_l)
        .def_readwrite("z_u", &piqp::Result<T>::z_u)
        .def_readwrite("z_bl", &piqp::Result<T>::z_bl)
        .def_readwrite("z_bu", &piqp::Result<T>::z_bu)
        .def_readwrite("s_l", &piqp::Result<T>::s_l)
        .def_readwrite("s_u", &piqp::Result<T>::s_u)
        .def_readwrite("s_bl", &piqp::Result<T>::s_bl)
        .def_readwrite("s_bu", &piqp::Result<T>::s_bu)
        .def_readwrite("info", &piqp::Result<T>::info);

    py::enum_<piqp::KKTSolver>(m, "KKTSolver", py::module_local())
            .value("dense_cholesky", piqp::KKTSolver::dense_cholesky)
            .value("sparse_ldlt", piqp::KKTSolver::sparse_ldlt)
            .value("sparse_ldlt_eq_cond", piqp::KKTSolver::sparse_ldlt_eq_cond)
            .value("sparse_ldlt_ineq_cond", piqp::KKTSolver::sparse_ldlt_ineq_cond)
            .value("sparse_ldlt_cond", piqp::KKTSolver::sparse_ldlt_cond)
            .value("sparse_multistage", piqp::KKTSolver::sparse_multistage)
            .value("sparse_multistage_parallel", piqp::KKTSolver::sparse_multistage_parallel)
            .export_values();

    py::class_<piqp::Settings<T>>(m, "Settings", py::module_local())
        .def_readwrite("rho_init", &piqp::Settings<T>::rho_init)
        .def_readwrite("delta_init", &piqp::Settings<T>::delta_init)
        .def_readwrite("eps_abs", &piqp::Settings<T>::eps_abs)
        .def_readwrite("eps_rel", &piqp::Settings<T>::eps_rel)
        .def_readwrite("check_duality_gap", &piqp::Settings<T>::check_duality_gap)
        .def_readwrite("eps_duality_gap_abs", &piqp::Settings<T>::eps_duality_gap_abs)
        .def_readwrite("eps_duality_gap_rel", &piqp::Settings<T>::eps_duality_gap_rel)
        .def_readwrite("infeasibility_threshold", &piqp::Settings<T>::infeasibility_threshold)
        .def_readwrite("eps_primal_inf", &piqp::Settings<T>::eps_primal_inf)
        .def_readwrite("eps_dual_inf", &piqp::Settings<T>::eps_dual_inf)
        .def_readwrite("reg_lower_limit", &piqp::Settings<T>::reg_lower_limit)
        .def_readwrite("reg_finetune_lower_limit", &piqp::Settings<T>::reg_finetune_lower_limit)
        .def_readwrite("reg_finetune_primal_update_threshold", &piqp::Settings<T>::reg_finetune_primal_update_threshold)
        .def_readwrite("reg_finetune_dual_update_threshold", &piqp::Settings<T>::reg_finetune_dual_update_threshold)
        .def_readwrite("max_iter", &piqp::Settings<T>::max_iter)
        .def_readwrite("max_factor_retires", &piqp::Settings<T>::max_factor_retires)
        .def_readwrite("preconditioner_scale_cost", &piqp::Settings<T>::preconditioner_scale_cost)
        .def_readwrite("preconditioner_reuse_on_update", &piqp::Settings<T>::preconditioner_reuse_on_update)
        .def_readwrite("preconditioner_iter", &piqp::Settings<T>::preconditioner_iter)
        .def_readwrite("tau", &piqp::Settings<T>::tau)
        .def_readwrite("kkt_solver", &piqp::Settings<T>::kkt_solver)
        .def_readwrite("num_threads", &piqp::Settings<T>::num_threads)
        .def_readwrite("iterative_refinement_always_enabled", &piqp::Settings<T>::iterative_refinement_always_enabled)
        .def_readwrite("iterative_refinement_eps_abs", &piqp::Settings<T>::iterative_refinement_eps_abs)
        .def_readwrite("iterative_refinement_eps_rel", &piqp::Settings<T>::iterative_refinement_eps_rel)
        .def_readwrite("iterative_refinement_max_iter", &piqp::Settings<T>::iterative_refinement_max_iter)
        .def_readwrite("iterative_refinement_min_improvement_rate", &piqp::Settings<T>::iterative_refinement_min_improvement_rate)
        .def_readwrite("iterative_refinement_static_regularization_eps", &piqp::Settings<T>::iterative_refinement_static_regularization_eps)
        .def_readwrite("iterative_refinement_static_regularization_rel", &piqp::Settings<T>::iterative_refinement_static_regularization_rel)
        .def_readwrite("enable_derivatives", &piqp::Settings<T>::enable_derivatives)
        .def_readwrite("derivative_smoothing", &piqp::Settings<T>::derivative_smoothing)
        .def_readwrite("derivative_smoothing_mu", &piqp::Settings<T>::derivative_smoothing_mu)
        .def_readwrite("derivative_smoothing_max_iter", &piqp::Settings<T>::derivative_smoothing_max_iter)
        .def_readwrite("verbose", &piqp::Settings<T>::verbose)
        .def_readwrite("compute_timings", &piqp::Settings<T>::compute_timings);

    using DenseVJPResult = piqp::dense::VJPResult<T>;
    py::class_<DenseVJPResult>(m, "DenseVJPResult", py::module_local())
        .def_readonly("status", &DenseVJPResult::status)
        .def_readonly("P", &DenseVJPResult::P)
        .def_readonly("c", &DenseVJPResult::c)
        .def_readonly("A", &DenseVJPResult::A)
        .def_readonly("b", &DenseVJPResult::b)
        .def_readonly("G", &DenseVJPResult::G)
        .def_readonly("h_l", &DenseVJPResult::h_l)
        .def_readonly("h_u", &DenseVJPResult::h_u)
        .def_readonly("x_l", &DenseVJPResult::x_l)
        .def_readonly("x_u", &DenseVJPResult::x_u);

    using SparseVJPResult = piqp::sparse::VJPResult<T, I>;
    py::class_<SparseVJPResult>(m, "SparseVJPResult", py::module_local())
        .def_readonly("status", &SparseVJPResult::status)
        .def_property_readonly("P", [](const py::object& self) { return sparse_matrix_view(self.cast<const SparseVJPResult&>().P, self); })
        .def_readonly("c", &SparseVJPResult::c)
        .def_property_readonly("A", [](const py::object& self) { return sparse_matrix_view(self.cast<const SparseVJPResult&>().A, self); })
        .def_readonly("b", &SparseVJPResult::b)
        .def_property_readonly("G", [](const py::object& self) { return sparse_matrix_view(self.cast<const SparseVJPResult&>().G, self); })
        .def_readonly("h_l", &SparseVJPResult::h_l)
        .def_readonly("h_u", &SparseVJPResult::h_u)
        .def_readonly("x_l", &SparseVJPResult::x_l)
        .def_readonly("x_u", &SparseVJPResult::x_u);

    using SparseSolver = piqp::SparseSolver<T, I>;
    py::class_<SparseSolver>(m, "SparseSolver", py::module_local())
        .def(py::init<>())
        .def_property("settings", &SparseSolver::settings, &SparseSolver::settings)
        .def_property_readonly("result", &SparseSolver::result)
        .def("setup",
             [](SparseSolver &solver,
                const piqp::SparseMat<T, I>& P,
                const piqp::CVecRef<T>& c,
                const piqp::optional<piqp::SparseMat<T, I>>& A = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& b = piqp::nullopt,
                const piqp::optional<piqp::SparseMat<T, I>>& G = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_u = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_u = piqp::nullopt)
             {
                 solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
             },
             py::arg("P"), py::arg("c"),
             py::arg("A") = piqp::nullopt, py::arg("b") = piqp::nullopt,
             py::arg("G") = piqp::nullopt, py::arg("h_l") = piqp::nullopt, py::arg("h_u") = piqp::nullopt,
             py::arg("x_l") = piqp::nullopt, py::arg("x_u") = piqp::nullopt)
        .def("update",
             [](SparseSolver &solver,
                const piqp::optional<piqp::SparseMat<T, I>>& P = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& c = piqp::nullopt,
                const piqp::optional<piqp::SparseMat<T, I>>& A = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& b = piqp::nullopt,
                const piqp::optional<piqp::SparseMat<T, I>>& G = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_u = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_u = piqp::nullopt)
             {
                 solver.update(P, c, A, b, G, h_l, h_u, x_l, x_u);
             },
             py::arg("P") = piqp::nullopt, py::arg("c") = piqp::nullopt,
             py::arg("A") = piqp::nullopt, py::arg("b") = piqp::nullopt,
             py::arg("G") = piqp::nullopt, py::arg("h_l") = piqp::nullopt, py::arg("h_u") = piqp::nullopt,
             py::arg("x_l") = piqp::nullopt, py::arg("x_u") = piqp::nullopt)
        .def("solve", &SparseSolver::solve)
        .def("vjp", &SparseSolver::vjp,
             py::arg("grad_x") = piqp::nullopt, py::arg("grad_y") = piqp::nullopt,
             py::arg("grad_z_l") = piqp::nullopt, py::arg("grad_z_u") = piqp::nullopt,
             py::arg("grad_z_bl") = piqp::nullopt, py::arg("grad_z_bu") = piqp::nullopt,
             py::arg("grad_s_l") = piqp::nullopt, py::arg("grad_s_u") = piqp::nullopt,
             py::arg("grad_s_bl") = piqp::nullopt, py::arg("grad_s_bu") = piqp::nullopt,
             py::arg("solve_result") = piqp::nullopt)
        .def_property_readonly("vjp_result", &SparseSolver::vjp_result);

    using DenseSolver = piqp::DenseSolver<T>;
    py::class_<DenseSolver>(m, "DenseSolver", py::module_local())
        .def(py::init<>())
        .def_property("settings", &DenseSolver::settings, &DenseSolver::settings)
        .def_property_readonly("result", &DenseSolver::result)
        .def("setup", &DenseSolver::setup,
             py::arg("P"), py::arg("c"),
             py::arg("A") = piqp::nullopt, py::arg("b") = piqp::nullopt,
             py::arg("G") = piqp::nullopt, py::arg("h_l") = piqp::nullopt, py::arg("h_u") = piqp::nullopt,
             py::arg("x_l") = piqp::nullopt, py::arg("x_u") = piqp::nullopt)
        .def("update", &DenseSolver::update,
             py::arg("P") = piqp::nullopt, py::arg("c") = piqp::nullopt,
             py::arg("A") = piqp::nullopt, py::arg("b") = piqp::nullopt,
             py::arg("G") = piqp::nullopt, py::arg("h_l") = piqp::nullopt, py::arg("h_u") = piqp::nullopt,
             py::arg("x_l") = piqp::nullopt, py::arg("x_u") = piqp::nullopt)
        .def("solve", &DenseSolver::solve)
        .def("vjp", &DenseSolver::vjp,
             py::arg("grad_x") = piqp::nullopt, py::arg("grad_y") = piqp::nullopt,
             py::arg("grad_z_l") = piqp::nullopt, py::arg("grad_z_u") = piqp::nullopt,
             py::arg("grad_z_bl") = piqp::nullopt, py::arg("grad_z_bu") = piqp::nullopt,
             py::arg("grad_s_l") = piqp::nullopt, py::arg("grad_s_u") = piqp::nullopt,
             py::arg("grad_s_bl") = piqp::nullopt, py::arg("grad_s_bu") = piqp::nullopt,
             py::arg("solve_result") = piqp::nullopt)
        .def_property_readonly("vjp_result", &DenseSolver::vjp_result);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
