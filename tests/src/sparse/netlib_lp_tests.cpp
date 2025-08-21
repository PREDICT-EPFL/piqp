// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
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

class SparseNetlibFeasibleTest : public testing::TestWithParam<std::string> {};
class SparseNetlibInfeasibleTest : public testing::TestWithParam<std::string> {};

TEST_P(SparseNetlibFeasibleTest, NetlibFeasible)
{
    std::string path = "data/netlib/data/" + GetParam();
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>(path);

    piqp::SparseSolver<T, I> solver;
    solver.settings().verbose = true;
    solver.settings().infeasibility_threshold = 0.01;
    solver.setup(model.P, model.c,
                 model.A, model.b,
                 model.G, model.h_l, model.h_u,
                 model.x_l, model.x_u);

    piqp::Status status = solver.solve();
    ASSERT_TRUE(status == piqp::Status::PIQP_SOLVED);
}

TEST_P(SparseNetlibInfeasibleTest, NetlibInfeasible)
{
    std::string path = "data/netlib/infeas/" + GetParam();
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>(path);

    piqp::SparseSolver<T, I> solver;
    solver.settings().verbose = true;
    solver.settings().infeasibility_threshold = 0.01;
    solver.setup(model.P, model.c,
                 model.A, model.b,
                 model.G, model.h_l, model.h_u,
                 model.x_l, model.x_u);

    piqp::Status status = solver.solve();
    ASSERT_TRUE(status == piqp::Status::PIQP_PRIMAL_INFEASIBLE || status == piqp::Status::PIQP_DUAL_INFEASIBLE);
}

std::vector<std::string> get_netlib_lp_feas_problems()
{
    std::vector<std::string> problem_names;
    for (const auto & entry : piqp::fs::directory_iterator("data/netlib/data"))
    {
        std::string file_name = entry.path().filename().string();
        if (file_name.substr(file_name.size() - 4) != ".mat") continue;

        problem_names.push_back(file_name);
    }
    return problem_names;
}

INSTANTIATE_TEST_SUITE_P(
    FromFolderData,
    SparseNetlibFeasibleTest,
    ::testing::ValuesIn(get_netlib_lp_feas_problems()),
//    ::testing::Values("25fv47.mat"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        piqp::usize i = info.param.find(".");
        std::string name = info.param.substr(0, i);
        std::replace(name.begin(), name.end(), '-', '_');
        return name;
    }
);

std::vector<std::string> get_netlib_lp_infeas_problems()
{
    std::vector<std::string> problem_names;
    for (const auto & entry : piqp::fs::directory_iterator("data/netlib/infeas"))
    {
        std::string file_name = entry.path().filename().string();
        if (file_name.substr(file_name.size() - 4) != ".mat") continue;

        problem_names.push_back(file_name);
    }
    return problem_names;
}

INSTANTIATE_TEST_SUITE_P(
    FromFolderInfeas,
    SparseNetlibInfeasibleTest,
    ::testing::ValuesIn(get_netlib_lp_infeas_problems()),
//    ::testing::Values("bgdbg1.mat"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        piqp::usize i = info.param.find(".");
        std::string name = info.param.substr(0, i);
        std::replace(name.begin(), name.end(), '-', '_');
        return name;
    }
);
