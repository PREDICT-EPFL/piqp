// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>

#include "piqp/piqp.hpp"
#include "piqp/utils/filesystem.hpp"
#include "piqp/utils/io_utils.hpp"

#include "gtest/gtest.h"

using T = double;
using I = int;

class SparseMarosMeszarosTest : public testing::TestWithParam<std::string> {};

TEST_P(SparseMarosMeszarosTest, CanSolveProblemKKTFull)
{
    std::string path = "maros_meszaros_data/" + GetParam();
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>(path);

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
    for (const auto & entry : piqp::fs::directory_iterator("maros_meszaros_data"))
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
