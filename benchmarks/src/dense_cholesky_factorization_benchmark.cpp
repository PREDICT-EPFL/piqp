// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include "piqp/dense/ldlt_no_pivot.hpp"
#include "piqp/utils/random_utils.hpp"

using namespace piqp;
using namespace piqp::dense;

template<typename T>
static void BM_EIGEN_LLT_LOWER(benchmark::State& state)
{
    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(state.range(0), 1.0);
    P.transposeInPlace();

    Eigen::LLT<Mat<T>, Eigen::Lower> llt(P.rows());

    for (auto _ : state)
    {
        llt.compute(P);
    }
}

template<typename T>
static void BM_EIGEN_LLT_UPPER(benchmark::State& state)
{
    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(state.range(0), 1.0);

    Eigen::LLT<Mat<T>, Eigen::Upper> llt(P.rows());

    for (auto _ : state)
    {
        llt.compute(P);
    }
}

template<typename T>
static void BM_EIGEN_LDLT_LOWER(benchmark::State& state)
{
    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(state.range(0), 1.0);
    P.transposeInPlace();

    Eigen::LDLT<Mat<T>, Eigen::Lower> ldlt(P.rows());

    for (auto _ : state)
    {
        ldlt.compute(P);
    }
}

template<typename T>
static void BM_EIGEN_LDLT_UPPER(benchmark::State& state)
{
    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(state.range(0), 1.0);

    Eigen::LDLT<Mat<T>, Eigen::Upper> ldlt(P.rows());

    for (auto _ : state)
    {
        ldlt.compute(P);
    }
}

template<typename T>
static void BM_PIQP_LDLT_NO_PIVOT_LOWER(benchmark::State& state)
{
    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(state.range(0), 1.0);
    P.transposeInPlace();

    LDLTNoPivot<Mat<T>, Eigen::Lower> ldlt(P.rows());

    for (auto _ : state)
    {
        ldlt.compute(P);
    }
}

template<typename T>
static void BM_PIQP_LDLT_NO_PIVOT_UPPER(benchmark::State& state)
{
    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(state.range(0), 1.0);

    LDLTNoPivot<Mat<T>, Eigen::Upper> ldlt(P.rows());

    for (auto _ : state)
    {
        ldlt.compute(P);
    }
}

BENCHMARK(BM_EIGEN_LLT_LOWER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EIGEN_LLT_UPPER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EIGEN_LDLT_LOWER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EIGEN_LDLT_UPPER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_PIQP_LDLT_NO_PIVOT_LOWER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_PIQP_LDLT_NO_PIVOT_UPPER<double>)->RangeMultiplier(2)->Range(4, 1<<10)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
