# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5)

# Google Benchmark
include(FetchContent)
FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/84c71faa8126e4eedc2bb520352615cb4484d6ad.zip
)
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
FetchContent_MakeAvailable(benchmark)

add_library(pipq-benchmark INTERFACE)
target_compile_options(pipq-benchmark INTERFACE ${compiler_flags})
target_link_options(pipq-benchmark INTERFACE ${compiler_flags})
target_link_libraries(pipq-benchmark INTERFACE piqp benchmark::benchmark)

add_executable(dense_cholesky_factorization_benchmark src/dense_cholesky_factorization_benchmark.cpp)
target_link_libraries(dense_cholesky_factorization_benchmark PRIVATE pipq-benchmark)
