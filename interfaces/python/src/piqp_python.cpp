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

using T = double;
using I = int;

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
    py::enum_<piqp::Status>(m, "Status")
        .value("PIQP_SOLVED", piqp::Status::PIQP_SOLVED)
        .value("PIQP_MAX_ITER_REACHED", piqp::Status::PIQP_MAX_ITER_REACHED)
        .value("PIQP_PRIMAL_INFEASIBLE", piqp::Status::PIQP_PRIMAL_INFEASIBLE)
        .value("PIQP_DUAL_INFEASIBLE", piqp::Status::PIQP_DUAL_INFEASIBLE)
        .value("PIQP_NUMERICS", piqp::Status::PIQP_NUMERICS)
        .value("PIQP_UNSOLVED", piqp::Status::PIQP_UNSOLVED)
        .value("PIQP_INVALID_SETTINGS", piqp::Status::PIQP_INVALID_SETTINGS)
        .export_values();

    py::class_<piqp::Info<T>>(m, "Info")
        .def(py::init<>())
        .def_readwrite("status", &piqp::Info<T>::status)
        .def_readwrite("iter", &piqp::Info<T>::iter)
        .def_readwrite("rho", &piqp::Info<T>::rho)
        .def_readwrite("delta", &piqp::Info<T>::delta)
        .def_readwrite("mu", &piqp::Info<T>::mu)
        .def_readwrite("sigma", &piqp::Info<T>::sigma)
        .def_readwrite("primal_step", &piqp::Info<T>::primal_step)
        .def_readwrite("dual_step", &piqp::Info<T>::dual_step)
        .def_readwrite("primal_inf", &piqp::Info<T>::primal_inf)
        .def_readwrite("dual_inf", &piqp::Info<T>::dual_inf)
        .def_readwrite("factor_retires", &piqp::Info<T>::factor_retires)
        .def_readwrite("reg_limit", &piqp::Info<T>::reg_limit)
        .def_readwrite("no_primal_update", &piqp::Info<T>::no_primal_update)
        .def_readwrite("no_dual_update", &piqp::Info<T>::no_dual_update)
        .def_readwrite("setup_time", &piqp::Info<T>::setup_time)
        .def_readwrite("update_time", &piqp::Info<T>::update_time)
        .def_readwrite("solve_time", &piqp::Info<T>::solve_time)
        .def_readwrite("run_time", &piqp::Info<T>::run_time);

    py::class_<piqp::Result<T>>(m, "Result")
        .def_readwrite("x", &piqp::Result<T>::x)
        .def_readwrite("y", &piqp::Result<T>::y)
        .def_readwrite("z", &piqp::Result<T>::z)
        .def_readwrite("z_lb", &piqp::Result<T>::z_lb)
        .def_readwrite("z_ub", &piqp::Result<T>::z_ub)
        .def_readwrite("s", &piqp::Result<T>::s)
        .def_readwrite("s_lb", &piqp::Result<T>::s_lb)
        .def_readwrite("s_ub", &piqp::Result<T>::s_ub)
        .def_readwrite("zeta", &piqp::Result<T>::zeta)
        .def_readwrite("lambda", &piqp::Result<T>::lambda)
        .def_readwrite("nu", &piqp::Result<T>::nu)
        .def_readwrite("nu_lb", &piqp::Result<T>::nu_lb)
        .def_readwrite("nu_ub", &piqp::Result<T>::nu_ub)
        .def_readwrite("info", &piqp::Result<T>::info);

    py::class_<piqp::Settings<T>>(m, "Settings")
        .def_readwrite("rho_init", &piqp::Settings<T>::rho_init)
        .def_readwrite("delta_init", &piqp::Settings<T>::delta_init)
        .def_readwrite("feas_tol_abs", &piqp::Settings<T>::feas_tol_abs)
        .def_readwrite("feas_tol_rel", &piqp::Settings<T>::feas_tol_rel)
        .def_readwrite("dual_tol", &piqp::Settings<T>::dual_tol)
        .def_readwrite("reg_lower_limit", &piqp::Settings<T>::reg_lower_limit)
        .def_readwrite("max_iter", &piqp::Settings<T>::max_iter)
        .def_readwrite("max_factor_retires", &piqp::Settings<T>::max_factor_retires)
        .def_readwrite("preconditioner_iter", &piqp::Settings<T>::preconditioner_iter)
        .def_readwrite("tau", &piqp::Settings<T>::tau)
        .def_readwrite("verbose", &piqp::Settings<T>::verbose)
        .def_readwrite("compute_timings", &piqp::Settings<T>::compute_timings);

    using SparseSolver = piqp::SparseSolver<T, I, piqp::KKTMode::KKT_FULL>;
    py::class_<SparseSolver>(m, "SparseSolver")
        .def(py::init<>())
        .def_property("settings", &SparseSolver::settings, &SparseSolver::settings)
        .def_property_readonly("result", &SparseSolver::result)
        .def("setup", &SparseSolver::setup)
        .def("update", &SparseSolver::update)
        .def("solve", &SparseSolver::solve);

    using DenseSolver = piqp::DenseSolver<T>;
    py::class_<DenseSolver>(m, "DenseSolver")
        .def(py::init<>())
        .def_property("settings", &DenseSolver::settings, &DenseSolver::settings)
        .def_property_readonly("result", &DenseSolver::result)
        .def("setup", &DenseSolver::setup)
        .def("update", &DenseSolver::update)
        .def("solve", &DenseSolver::solve);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
