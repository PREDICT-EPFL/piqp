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
    sparse::Model<T, I> generate_random_multistage_qp(int N, int diag_block_size = 50)
    {
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

        SparseMat<T, I> A(m_total, n_total);
        A.setFromTriplets(triplets.begin(), triplets.end());

        SparseMat<T, I> P(n_total, n_total);
        P.setIdentity();

        Vec<T> c = Vec<T>::Zero(n_total);
        Vec<T> b = A * Vec<T>::Ones(n_total);

        SparseMat<T, I> G(0, n_total);
        Vec<T> h_l = Vec<T>::Zero(0);
        Vec<T> h_u = Vec<T>::Zero(0);

        Vec<T> x_l = Vec<T>::Constant(n_total, -10);
        Vec<T> x_u = Vec<T>::Constant(n_total, 10);

        return sparse::Model<T, I>(P, c, A, b, G, h_l, h_u, x_l, x_u);
    }
} // namespace piqp

static std::vector<int64_t> thread_counts()
{
    // 0 uses the OpenMP default, i.e., omp_get_max_threads()
    std::vector<int64_t> counts = {0};
#ifdef PIQP_HAS_OPENMP
    const int max_threads = omp_get_max_threads();
    for (int t : {1, 2, 4, 6, 8, 12, 16}) {
        if (t < max_threads) counts.push_back(t);
    }
    counts.push_back(max_threads);
#endif
    return counts;
}

static void run_multistage_benchmark(benchmark::State& state, const piqp::sparse::Model<T, I>& model,
                                     piqp::KKTSolver kkt_solver, piqp::isize num_threads)
{
    piqp::SparseSolver<T, I> solver;
    solver.settings().kkt_solver = kkt_solver;
    solver.settings().num_threads = num_threads;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    piqp::Status status = piqp::Status::PIQP_UNSOLVED;
    for (auto _ : state)
    {
        status = solver.solve();
    }

    state.counters["iter"] = static_cast<double>(solver.result().info.iter);
    state.SetLabel(piqp::status_to_string(status));
}

// Arguments: {threads}
static void thread_args(benchmark::internal::Benchmark* b)
{
    for (int64_t t : thread_counts()) b->Args({t});
}

// Robot arm and chain mass have too few stages to be representative for the
// parallel backend, uncomment to include them.
// static void BM_ROBOT_ARM_SQP_MULTISTAGE_KKT(benchmark::State& state)
// {
//     piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/robot_arm_sqp.mat");
//     run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage, state.range(0));
// }
//
// static void BM_ROBOT_ARM_SQP_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
// {
//     piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/robot_arm_sqp.mat");
//     run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage_parallel, state.range(0));
// }
//
// static void BM_CHAIN_MASS_SQP_MULTISTAGE_KKT(benchmark::State& state)
// {
//     piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/chain_mass_sqp.mat");
//     run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage, state.range(0));
// }
//
// static void BM_CHAIN_MASS_SQP_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
// {
//     piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/chain_mass_sqp.mat");
//     run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage_parallel, state.range(0));
// }

static void BM_RACE_LINE_MULTISTAGE_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/race_line.mat");
    run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage, state.range(0));
}

static void BM_RACE_LINE_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/race_line.mat");
    run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage_parallel, state.range(0));
}

// Arguments: {N, threads}
static void random_args(benchmark::internal::Benchmark* b)
{
    for (int64_t N : {50, 100, 200, 500, 1000}) {
        for (int64_t t : thread_counts()) b->Args({N, t});
    }
}

static void BM_RANDOM_MULTISTAGE_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::generate_random_multistage_qp<T, I>(static_cast<int>(state.range(0)));
    run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage, state.range(1));
}

static void BM_RANDOM_MULTISTAGE_PARALLEL_KKT(benchmark::State& state)
{
    piqp::sparse::Model<T, I> model = piqp::generate_random_multistage_qp<T, I>(static_cast<int>(state.range(0)));
    run_multistage_benchmark(state, model, piqp::KKTSolver::sparse_multistage_parallel, state.range(1));
}

// BENCHMARK(BM_ROBOT_ARM_SQP_MULTISTAGE_KKT)->Apply(thread_args)->ArgNames({"threads"})->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_ROBOT_ARM_SQP_MULTISTAGE_PARALLEL_KKT)->Apply(thread_args)->ArgNames({"threads"})->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_CHAIN_MASS_SQP_MULTISTAGE_KKT)->Apply(thread_args)->ArgNames({"threads"})->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_CHAIN_MASS_SQP_MULTISTAGE_PARALLEL_KKT)->Apply(thread_args)->ArgNames({"threads"})->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RACE_LINE_MULTISTAGE_KKT)->Apply(thread_args)->ArgNames({"threads"})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RACE_LINE_MULTISTAGE_PARALLEL_KKT)->Apply(thread_args)->ArgNames({"threads"})->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RANDOM_MULTISTAGE_KKT)->Apply(random_args)->ArgNames({"N", "threads"})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RANDOM_MULTISTAGE_PARALLEL_KKT)->Apply(random_args)->ArgNames({"N", "threads"})->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
