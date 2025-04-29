// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"

using namespace piqp;

template<typename T>
static void BM_DENSE_SOLVER(benchmark::State& state)
{
    isize dim = state.range(0);
    isize n_eq = dim / 2;
    isize n_ineq = dim / 2;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);

    DenseSolver<T> solver;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    for (auto _ : state)
    {
        solver.solve();
    }
}

template<typename T, typename I>
static void BM_SPARSE_SOLVER(benchmark::State& state)
{
    isize dim = state.range(0);
    isize n_eq = dim / 2;
    isize n_ineq = dim / 2;
    T sparsity_factor = 0.1;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    SparseSolver<T, I> solver;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    for (auto _ : state)
    {
        solver.solve();
    }
}

BENCHMARK(BM_DENSE_SOLVER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SPARSE_SOLVER<double, int>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
