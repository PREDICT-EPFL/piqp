// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <limits>
#include "piqp/piqp.hpp"

int main()
{
    int n = 2;
    int p = 1;
    int m = 2;

    Eigen::MatrixXd P(n, n); P << 6, 0, 0, 4;
    Eigen::VectorXd c(n); c << -1, -4;

    Eigen::MatrixXd A(p, n); A << 1, -2;
    Eigen::VectorXd b(p); b << 1;

    Eigen::MatrixXd G(m, n); G << 1, -1, 2, 0;
    Eigen::VectorXd h_u(m); h_u << 0.2, -1;

    Eigen::VectorXd x_l(n); x_l << -1, -std::numeric_limits<double>::infinity();
    Eigen::VectorXd x_u(n); x_u << 1, std::numeric_limits<double>::infinity();

    piqp::DenseSolver<double> solver;
    solver.settings().verbose = true;
    solver.settings().compute_timings = true;
    solver.setup(P, c, A, b, G, piqp::nullopt, h_u, x_l, x_u);

    piqp::Status status = solver.solve();

    std::cout << "status = " << status << std::endl;
    std::cout << "x = " << solver.result().x.transpose() << std::endl;

    return 0;
}
