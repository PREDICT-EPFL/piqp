// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include <random>
#include "piqp/piqp.hpp"
#include "piqp/utils/io_utils.hpp"

using T = double;
using I = int;

namespace piqp
{
    template<typename T, typename I>
    void generate_random_multistage_qp_data(
            SparseMat<T, I>& P,
            Vec<T>& c,
            SparseMat<T, I>& A,
            Vec<T>& b,
            SparseMat<T, I>& G,
            Vec<T>& h_l,
            Vec<T>& h_u,
            Vec<T>& x_l,
            Vec<T>& x_u) {
        static bool initialized = false;

        static SparseMat<T, I> A_static;
        static Vec<T> b_static;
        static SparseMat<T, I> P_static;
        static Vec<T> c_static;
        static SparseMat<T, I> G_static;
        static Vec<T> h_l_static, h_u_static;
        static Vec<T> x_l_static, x_u_static;

        if (!initialized) {
            const int N = 100;
            const int diag_block_size = 50;
            const int offdiag_block_size = diag_block_size / 2;

            const int n_total = N * diag_block_size;
            const int m_total = N * offdiag_block_size;

            std::mt19937 gen(42);
            std::normal_distribution<T> dist(0.0, 1.0);

            std::vector<Eigen::Triplet<T>> triplets;

            for (int i = 0; i < N; ++i) {
                int row_start = offdiag_block_size * i;
                int col_start = diag_block_size * i;
                for (int row = 0; row < offdiag_block_size; ++row) {
                    for (int col = 0; col < diag_block_size; ++col) {
                        triplets.emplace_back(row_start + row, col_start + col, dist(gen));
                    }
                }
            }

            for (int i = 0; i < N - 1; ++i) {
                int row_start = offdiag_block_size * (i + 1);
                int col_start = diag_block_size * i;
                for (int row = 0; row < offdiag_block_size; ++row) {
                    for (int col = 0; col < diag_block_size; ++col) {
                        triplets.emplace_back(row_start + row, col_start + col, dist(gen));
                    }
                }
            }

            A_static.resize(m_total, n_total);
            A_static.setFromTriplets(triplets.begin(), triplets.end());

            P_static.resize(n_total, n_total);
            P_static.setIdentity();

            c_static = Vec<T>::Zero(n_total);
            h_l_static = Vec<T>::Zero(0, 1);
            h_u_static = Vec<T>::Zero(0, 1);
            G_static.resize(0, n_total);

            x_l_static = Vec<T>::Constant(n_total, -1e8);
            x_u_static = Vec<T>::Constant(n_total,  1e8);

            Vec<T> ones = Vec<T>::Ones(n_total);
            b_static = A_static * ones;

            initialized = true;
        }

        // Copy out
        A     = A_static;
        b     = b_static;
        P     = P_static;
        c     = c_static;
        G     = G_static;
        h_l   = h_l_static;
        h_u   = h_u_static;
        x_l   = x_l_static;
        x_u   = x_u_static;
    }
} // namespace piqp

static void BM_ROBOT_ARM_SQP_MULTISTAGE_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/robot_arm_sqp.mat");
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    for (auto _ : state)
    {
        // solver.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
        solver.solve();
    }
}


static void BM_ROBOT_ARM_SQP_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/robot_arm_sqp.mat");
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage_parallel;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    for (auto _ : state)
    {
        // solver.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
        solver.solve();
    }
}


static void BM_CHAIN_MASS_SQP_MULTISTAGE_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/chain_mass_sqp.mat");
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    for (auto _ : state)
    {
        // solver.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
        solver.solve();
    }
}


static void BM_CHAIN_MASS_SQP_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/chain_mass_sqp.mat");
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage_parallel;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    for (auto _ : state)
    {
        // solver.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
        solver.solve();
    }
}


static void BM_RANDOM_SQP_MULTISTAGE_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/chain_mass_sqp.mat");
    piqp::generate_random_multistage_qp_data<T, I>(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    solver.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    for (auto _ : state)
    {
        solver.solve();
    }
}


static void BM_RANDOM_SQP_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/chain_mass_sqp.mat");
    piqp::generate_random_multistage_qp_data<T, I>(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = piqp::KKTSolver::sparse_multistage_parallel;
    solver.settings().verbose = false;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    solver.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    for (auto _ : state)
    {
        solver.solve();
    }
}

BENCHMARK(BM_ROBOT_ARM_SQP_MULTISTAGE_KKT)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ROBOT_ARM_SQP_MULTISTAGE_PARALLEL_KKT)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CHAIN_MASS_SQP_MULTISTAGE_KKT)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CHAIN_MASS_SQP_MULTISTAGE_PARALLEL_KKT)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RANDOM_SQP_MULTISTAGE_KKT)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RANDOM_SQP_MULTISTAGE_PARALLEL_KKT)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();