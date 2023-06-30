// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <filesystem>

#include <pybind11/embed.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/piqp.hpp"

#include "gtest/gtest.h"

using T = double;
using I = int;

namespace fs = std::filesystem;
namespace py = pybind11;
using namespace py::literals;

py::scoped_interpreter guard{};

piqp::sparse::Model<T, I> get_problem_data(const std::string& file_name)
{
    py::dict locals("file_name"_a = "maros_meszaros_data/" + file_name);
    py::exec(R"(
        import numpy as np
        import scipy.sparse as spa
        import scipy.io as spio

        m = spio.loadmat(file_name)

        # Convert matrices
        P = m['P'].astype(float).tocsc()
        q = m['q'].T.flatten().astype(float)
        r = m['r'].T.flatten().astype(float)[0]
        A = m['A'].astype(float).tocsc()
        l = m['l'].T.flatten().astype(float)
        u = m['u'].T.flatten().astype(float)
        n = m['n'].T.flatten().astype(int)[0]
        m = m['m'].T.flatten().astype(int)[0]

        l_inf = l
        u_inf = u
        l_inf[l_inf > +9e19] = +np.inf
        u_inf[u_inf > +9e19] = +np.inf
        l_inf[l_inf < -9e19] = -np.inf
        u_inf[u_inf < -9e19] = -np.inf

        # A == vstack([C, spa.eye(n)])
        xl = l_inf[-n:]
        xu = u_inf[-n:]
        C = A[:-n]
        cl = l_inf[:-n]
        cu = u_inf[:-n]

        # xl = np.ones(n) * -np.inf
        # xu = np.ones(n) * np.inf
        # C = A
        # cl = l_inf
        # cu = u_inf

        eq_bounds = cu == cl
        ineq_bounds = cu != cl

        eq_rows = np.asarray(eq_bounds).nonzero()
        A = C[eq_rows]
        b = cu[eq_rows]

        ineq_rows = np.asarray(ineq_bounds).nonzero()
        G = spa.vstack([C[ineq_rows], -C[ineq_rows]], format="csc")
        h = np.hstack([cu[ineq_rows], -cl[ineq_rows]])
        h_finite = h < np.inf
        if not h_finite.all():
            G = G[h_finite]
            h = h[h_finite]

        # csc matrices are not guaranteed to be sorted
        P.sort_indices()
        A.sort_indices()
        G.sort_indices()
    )", py::globals(), locals);

    auto P = locals["P"].cast<piqp::SparseMat<T, I>>();
    auto A = locals["A"].cast<piqp::SparseMat<T, I>>();
    auto G = locals["G"].cast<piqp::SparseMat<T, I>>();

    auto c = locals["q"].cast<piqp::Vec<T>>();
    auto b = locals["b"].cast<piqp::Vec<T>>();
    auto h = locals["h"].cast<piqp::Vec<T>>();
    auto x_lb = locals["xl"].cast<piqp::Vec<T>>();
    auto x_ub = locals["xu"].cast<piqp::Vec<T>>();

    return piqp::sparse::Model<T, I>(P, A, G, c, b, h, x_lb, x_ub);
}

class SparseMarosMeszarosTest : public testing::TestWithParam<std::string> {};

TEST_P(SparseMarosMeszarosTest, CanSolveProblemKKTFull)
{
    piqp::sparse::Model<T, I> model = get_problem_data(GetParam());

    piqp::SparseSolver<T, I> solver;
    solver.settings().verbose = true;
    solver.setup(model.P, model.c,
                 model.A, model.b,
                 model.G, model.h,
                 model.x_lb, model.x_ub);

    piqp::Status status = solver.solve();
    ASSERT_EQ(status, piqp::Status::PIQP_SOLVED);
}

std::vector<std::string> get_maros_meszaros_problems()
{
    std::vector<std::string> problem_names;
    for (const auto & entry : fs::directory_iterator("maros_meszaros_data"))
    {
        std::string file_name = entry.path().filename().string();
        if (file_name == "README.md" || file_name == "LICENSE") continue;

        problem_names.push_back(file_name);
    }
    return problem_names;
}

INSTANTIATE_TEST_SUITE_P(
    FromFolder,
    SparseMarosMeszarosTest,
    ::testing::ValuesIn(get_maros_meszaros_problems()),
//    ::testing::Values("QPILOTNO"),
//    ::testing::Values("QCAPRI"),
//    ::testing::Values("QBRANDY"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        piqp::usize i = info.param.find(".");
        std::string name = info.param.substr(0, i);
        std::replace(name.begin(), name.end(), '-', '_');
        return name;
    }
);
