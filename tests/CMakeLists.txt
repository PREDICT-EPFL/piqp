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

include_directories(include)

add_executable(kkt_condensed_test src/kkt_condensed_test.cpp)
target_link_libraries(kkt_condensed_test PRIVATE Eigen3.3 gtest_main gtest)

add_executable(ldlt_test src/ldlt_test.cpp)
target_link_libraries(ldlt_test PRIVATE Eigen3.3 gtest_main gtest)

include(GoogleTest)
gtest_discover_tests(kkt_condensed_test)
gtest_discover_tests(ldlt_test)
