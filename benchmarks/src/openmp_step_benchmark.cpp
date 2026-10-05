// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

// Measures when parallelizing the step length computation in
// SolverBase::calculate_step pays off. The kernel below mirrors calculate_step:
// a single parallel region containing three worksharing loops over the general
// inequalities (m) and the lower/upper box constraints (n_x_l, n_x_u).
//
// Benchmark arguments are {total, threads} with total = m + n_x_l + n_x_u,
// split as m = total / 2 and n_x_l = n_x_u = total / 4. threads == 0 runs the
// serial loops without opening a parallel region (the baseline), and
// threads == -1 uses the thread count chosen by resolve_elementwise_num_threads,
// i.e., what the solver does with the default settings.
//
// Compare real_time of a given total across thread counts. The elems_per_thread
// counter at the break-even point gives the minimum chunk size per thread
// (min_elementwise_elems_per_thread in piqp/utils/openmp.hpp). With a well tuned
// value, threads:-1 should be close to the fastest run for every total.
//
// Example:
//   ./openmp_step_benchmark --benchmark_counters_tabular=true

#include <benchmark/benchmark.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include "piqp/typedefs.hpp"
#include "piqp/utils/openmp.hpp"

using T = double;

namespace
{

struct StepData
{
    piqp::isize m, n_x_l, n_x_u;
    piqp::Vec<T> s_l, s_u, z_l, z_u, s_bl, s_bu, z_bl, z_bu;
    piqp::Vec<T> ds_l, ds_u, dz_l, dz_u, ds_bl, ds_bu, dz_bl, dz_bu;

    explicit StepData(piqp::isize total)
        : m(total / 2), n_x_l(total / 4), n_x_u(total - total / 2 - total / 4)
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> pos(0.1, 1.0);
        std::uniform_real_distribution<T> step(-1.0, 1.0);
        auto fill = [&gen](piqp::Vec<T>& v, piqp::isize n, std::uniform_real_distribution<T>& dist) {
            v.resize(n);
            for (piqp::isize i = 0; i < n; i++) v(i) = dist(gen);
        };
        fill(s_l, m, pos); fill(s_u, m, pos); fill(z_l, m, pos); fill(z_u, m, pos);
        fill(s_bl, n_x_l, pos); fill(z_bl, n_x_l, pos);
        fill(s_bu, n_x_u, pos); fill(z_bu, n_x_u, pos);
        fill(ds_l, m, step); fill(ds_u, m, step); fill(dz_l, m, step); fill(dz_u, m, step);
        fill(ds_bl, n_x_l, step); fill(dz_bl, n_x_l, step);
        fill(ds_bu, n_x_u, step); fill(dz_bu, n_x_u, step);
    }
};

inline void step_loops(const StepData& d, T& alpha_s, T& alpha_z)
{
    for (piqp::isize i = 0; i < d.m; i++)
    {
        if (d.ds_l(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_l(i) / d.ds_l(i));
        if (d.ds_u(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_u(i) / d.ds_u(i));
        if (d.dz_l(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_l(i) / d.dz_l(i));
        if (d.dz_u(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_u(i) / d.dz_u(i));
    }
    for (piqp::isize i = 0; i < d.n_x_l; i++)
    {
        if (d.ds_bl(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_bl(i) / d.ds_bl(i));
        if (d.dz_bl(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_bl(i) / d.dz_bl(i));
    }
    for (piqp::isize i = 0; i < d.n_x_u; i++)
    {
        if (d.ds_bu(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_bu(i) / d.ds_bu(i));
        if (d.dz_bu(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_bu(i) / d.dz_bu(i));
    }
}

inline void step_loops_parallel(const StepData& d, int num_threads, T& alpha_s, T& alpha_z)
{
#pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for reduction(min:alpha_s,alpha_z)
        for (piqp::isize i = 0; i < d.m; i++)
        {
            if (d.ds_l(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_l(i) / d.ds_l(i));
            if (d.ds_u(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_u(i) / d.ds_u(i));
            if (d.dz_l(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_l(i) / d.dz_l(i));
            if (d.dz_u(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_u(i) / d.dz_u(i));
        }
        #pragma omp for reduction(min:alpha_s,alpha_z)
        for (piqp::isize i = 0; i < d.n_x_l; i++)
        {
            if (d.ds_bl(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_bl(i) / d.ds_bl(i));
            if (d.dz_bl(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_bl(i) / d.dz_bl(i));
        }
        #pragma omp for reduction(min:alpha_s,alpha_z)
        for (piqp::isize i = 0; i < d.n_x_u; i++)
        {
            if (d.ds_bu(i) < 0) alpha_s = (std::min)(alpha_s, -d.s_bu(i) / d.ds_bu(i));
            if (d.dz_bu(i) < 0) alpha_z = (std::min)(alpha_z, -d.z_bu(i) / d.dz_bu(i));
        }
    }
}

} // namespace

static void BM_CALCULATE_STEP(benchmark::State& state)
{
    const piqp::isize total = state.range(0);
    int num_threads = static_cast<int>(state.range(1));
    if (num_threads < 0) {
        num_threads = piqp::resolve_elementwise_num_threads(0, total);
        if (num_threads == 1) num_threads = 0;
    }
    StepData data(total);

    for (auto _ : state)
    {
        T alpha_s = T(1);
        T alpha_z = T(1);
        if (num_threads == 0) {
            step_loops(data, alpha_s, alpha_z);
        } else {
            step_loops_parallel(data, num_threads, alpha_s, alpha_z);
        }
        benchmark::DoNotOptimize(alpha_s);
        benchmark::DoNotOptimize(alpha_z);
    }

    state.SetItemsProcessed(state.iterations() * total);
    state.counters["elems_per_thread"] = static_cast<double>(total) / (std::max)(1, num_threads);
}

// Same as BM_CALCULATE_STEP, but every iteration first runs a parallel region on
// all threads, mimicking the multistage parallel KKT backend which runs its own
// parallel regions right before the step length computation. Only the step
// length computation is timed, so this shows whether recently used worker
// threads make the parallel step cheaper (depends on KMP_BLOCKTIME, GOMP_SPINCOUNT, ...).
static void BM_CALCULATE_STEP_AFTER_PARALLEL_REGION(benchmark::State& state)
{
    const piqp::isize total = state.range(0);
    int num_threads = static_cast<int>(state.range(1));
    if (num_threads < 0) {
        num_threads = piqp::resolve_elementwise_num_threads(0, total);
        if (num_threads == 1) num_threads = 0;
    }
    StepData data(total);

    const int max_threads = omp_get_max_threads();
    std::vector<piqp::Vec<T>> kkt_work(static_cast<std::size_t>(max_threads), piqp::Vec<T>::Ones(1024));

    for (auto _ : state)
    {
#pragma omp parallel num_threads(max_threads)
        {
            piqp::Vec<T>& w = kkt_work[static_cast<std::size_t>(omp_get_thread_num())];
            w.array() = w.array() * T(0.5) + T(0.5);
            benchmark::DoNotOptimize(w.data());
        }

        auto start = std::chrono::steady_clock::now();
        T alpha_s = T(1);
        T alpha_z = T(1);
        if (num_threads == 0) {
            step_loops(data, alpha_s, alpha_z);
        } else {
            step_loops_parallel(data, num_threads, alpha_s, alpha_z);
        }
        benchmark::DoNotOptimize(alpha_s);
        benchmark::DoNotOptimize(alpha_z);
        auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(end - start).count());
    }

    state.SetItemsProcessed(state.iterations() * total);
    state.counters["elems_per_thread"] = static_cast<double>(total) / (std::max)(1, num_threads);
}

static void calculate_step_args(benchmark::internal::Benchmark* b)
{
    const int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts = {-1, 0};
    for (int t = 1; t < max_threads; t *= 2) thread_counts.push_back(t);
    thread_counts.push_back(max_threads);

    for (piqp::isize total = 64; total <= (1 << 20); total *= 2) {
        for (int t : thread_counts) {
            b->Args({total, t});
        }
    }
}

BENCHMARK(BM_CALCULATE_STEP)
    ->Apply(calculate_step_args)
    ->ArgNames({"total", "threads"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CALCULATE_STEP_AFTER_PARALLEL_REGION)
    ->Apply(calculate_step_args)
    ->ArgNames({"total", "threads"})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
