# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5)

# Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/86add13493e5c881d7e4ba77fb91c1f57752b3a4.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(pipq-test INTERFACE)
target_include_directories(pipq-test INTERFACE include)
target_compile_options(pipq-test INTERFACE ${compiler_flags})
target_link_options(pipq-test INTERFACE ${compiler_flags})
target_link_libraries(pipq-test INTERFACE piqp gtest_main gtest)

add_executable(dense_ldlt_test src/dense/ldlt_test.cpp)
target_link_libraries(dense_ldlt_test PRIVATE pipq-test)

add_executable(sparse_kkt_test src/sparse/kkt_test.cpp)
target_link_libraries(sparse_kkt_test PRIVATE pipq-test)

add_executable(sparse_ldlt_test src/sparse/ldlt_test.cpp)
target_link_libraries(sparse_ldlt_test PRIVATE pipq-test)

add_executable(sparse_utils_test src/sparse/utils_test.cpp)
target_link_libraries(sparse_utils_test PRIVATE pipq-test)

add_executable(solver_test src/solver_test.cpp)
target_link_libraries(solver_test PRIVATE pipq-test)

include(GoogleTest)
gtest_discover_tests(dense_ldlt_test)
gtest_discover_tests(sparse_kkt_test)
gtest_discover_tests(sparse_ldlt_test)
gtest_discover_tests(sparse_utils_test)
gtest_discover_tests(solver_test)
