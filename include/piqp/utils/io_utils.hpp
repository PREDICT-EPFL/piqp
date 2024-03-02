// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_IO_UTILS_HPP
#define PIQP_UTILS_IO_UTILS_HPP

#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>

#include "piqp/dense/model.hpp"
#include "piqp/sparse/model.hpp"

namespace piqp
{

static bool py_interpreter_initialized = false;
static void cleanup_py_interpreter()
{
    namespace py = pybind11;
    if (py_interpreter_initialized) {
        py::finalize_interpreter();
        py_interpreter_initialized = false;
    }
}

template<typename T>
void save_dense_model(const dense::Model<T>& model, const std::string& path)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    if (!py_interpreter_initialized) {
        py::initialize_interpreter();
        py_interpreter_initialized = true;
        std::atexit(cleanup_py_interpreter);
    }

    py::str py_path = path;
    pybind11::dict data(
        "P"_a = model.P,
        "A"_a = model.A,
        "G"_a = model.G,
        "c"_a = model.c,
        "b"_a = model.b,
        "h"_a = model.h,
        "x_lb"_a = model.x_lb,
        "x_ub"_a = model.x_ub
    );

    py::object spio = py::module_::import("scipy.io");
    spio.attr("savemat")(py_path, data);
}

template<typename T, typename I>
void save_sparse_model(const sparse::Model<T, I>& model, const std::string& path)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    if (!py_interpreter_initialized) {
        py::initialize_interpreter();
        py_interpreter_initialized = true;
        std::atexit(cleanup_py_interpreter);
    }

    py::str py_path = path;
    pybind11::dict data(
        "P"_a = model.P,
        "A"_a = model.A,
        "G"_a = model.G,
        "c"_a = model.c,
        "b"_a = model.b,
        "h"_a = model.h,
        "x_lb"_a = model.x_lb,
        "x_ub"_a = model.x_ub
    );

    py::object spio = py::module_::import("scipy.io");
    spio.attr("savemat")(py_path, data);
}

template<typename T>
dense::Model<T> load_dense_model(const std::string& path)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    if (!py_interpreter_initialized) {
        py::initialize_interpreter();
        py_interpreter_initialized = true;
        std::atexit(cleanup_py_interpreter);
    }

    py::object spio = py::module_::import("scipy.io");

    py::str py_path = path;
    pybind11::dict data = spio.attr("loadmat")(py_path);

    dense::Model<T> model(
        data["P"].cast<Mat<T>>(), data["c"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["A"].cast<Mat<T>>(), data["b"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["G"].cast<Mat<T>>(), data["h"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["x_lb"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["x_ub"].attr("T").attr("flatten")().cast<Vec<T>>()
    );

    return model;
}

template<typename T, typename I>
sparse::Model<T, I> load_sparse_model(const std::string& path)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    if (!py_interpreter_initialized) {
        py::initialize_interpreter();
        py_interpreter_initialized = true;
        std::atexit(cleanup_py_interpreter);
    }

    py::object spio = py::module_::import("scipy.io");

    py::str py_path = path;
    pybind11::dict data = spio.attr("loadmat")(py_path);

    sparse::Model<T, I> model(
        data["P"].cast<SparseMat<T, I>>(), data["c"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["A"].cast<SparseMat<T, I>>(), data["b"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["G"].cast<SparseMat<T, I>>(), data["h"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["x_lb"].attr("T").attr("flatten")().cast<Vec<T>>(),
        data["x_ub"].attr("T").attr("flatten")().cast<Vec<T>>()
    );

    return model;
}

} // namespace piqp

#endif //PIQP_UTILS_IO_UTILS_HPP
