# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/86add13493e5c881d7e4ba77fb91c1f57752b3a4.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

enable_testing()

# copy data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_library(piqp-test INTERFACE)
target_include_directories(piqp-test INTERFACE include)
target_compile_options(piqp-test INTERFACE ${compiler_flags})
if(${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
    target_compile_options(piqp-test INTERFACE -Wno-gnu-zero-variadic-macro-arguments)
endif()
target_link_options(piqp-test INTERFACE ${compiler_flags})
target_link_libraries(piqp-test INTERFACE piqp gtest_main gtest)

add_executable(dense_ldlt_test src/dense/ldlt_test.cpp)
target_link_libraries(dense_ldlt_test PRIVATE piqp-test)

add_executable(dense_kkt_test src/dense/kkt_test.cpp)
target_link_libraries(dense_kkt_test PRIVATE piqp-test)

add_executable(dense_solver_test src/dense/solver_test.cpp)
target_link_libraries(dense_solver_test PRIVATE piqp-test)

add_executable(multistage_kkt_test src/sparse/multistage_kkt_test.cpp)
target_link_libraries(multistage_kkt_test PRIVATE piqp-test Matio::Matio)

add_executable(sparse_kkt_test src/sparse/kkt_test.cpp)
target_link_libraries(sparse_kkt_test PRIVATE piqp-test)

add_executable(sparse_ldlt_test src/sparse/ldlt_test.cpp)
target_link_libraries(sparse_ldlt_test PRIVATE piqp-test)

add_executable(sparse_utils_test src/sparse/utils_test.cpp)
target_link_libraries(sparse_utils_test PRIVATE piqp-test)

add_executable(sparse_solver_test src/sparse/solver_test.cpp)
target_link_libraries(sparse_solver_test PRIVATE piqp-test)

add_executable(io_utils_test src/io_utils_test.cpp)
target_link_libraries(io_utils_test PRIVATE piqp-test Matio::Matio)

add_executable(preconditioner_test src/preconditioner_test.cpp)
target_link_libraries(preconditioner_test PRIVATE piqp-test)

if (BUILD_MAROS_MESZAROS_TESTS)
    add_executable(dense_maros_meszaros_tests src/dense/maros_meszaros_tests.cpp)
    target_link_libraries(dense_maros_meszaros_tests PRIVATE piqp-test Matio::Matio)

    add_executable(sparse_maros_meszaros_tests src/sparse/maros_meszaros_tests.cpp)
    target_link_libraries(sparse_maros_meszaros_tests PRIVATE piqp-test Matio::Matio)
endif()

if (BUILD_NETLIB_TESTS)
    add_executable(sparse_netlib_lp_tests src/sparse/netlib_lp_tests.cpp)
    target_link_libraries(sparse_netlib_lp_tests PRIVATE piqp-test Matio::Matio)
endif()

fix_test_dll(dense_ldlt_test)
fix_test_dll(dense_kkt_test)
fix_test_dll(dense_solver_test)
fix_test_dll(multistage_kkt_test)
fix_test_dll(sparse_kkt_test)
fix_test_dll(sparse_ldlt_test)
fix_test_dll(sparse_utils_test)
fix_test_dll(sparse_solver_test)
fix_test_dll(preconditioner_test)
fix_test_dll(io_utils_test)
if (BUILD_MAROS_MESZAROS_TESTS)
    fix_test_dll(dense_maros_meszaros_tests)
    fix_test_dll(sparse_maros_meszaros_tests)
endif()
if (BUILD_NETLIB_TESTS)
    fix_test_dll(sparse_netlib_lp_tests)
endif()

include(GoogleTest)
gtest_discover_tests(dense_ldlt_test)
gtest_discover_tests(dense_kkt_test)
gtest_discover_tests(dense_solver_test)
gtest_discover_tests(multistage_kkt_test)
gtest_discover_tests(sparse_kkt_test)
gtest_discover_tests(sparse_ldlt_test)
gtest_discover_tests(sparse_utils_test)
gtest_discover_tests(sparse_solver_test)
gtest_discover_tests(preconditioner_test)
gtest_discover_tests(io_utils_test)
if (BUILD_MAROS_MESZAROS_TESTS)
    gtest_discover_tests(dense_maros_meszaros_tests)
    gtest_discover_tests(sparse_maros_meszaros_tests)
endif()
if (BUILD_NETLIB_TESTS)
    gtest_discover_tests(sparse_netlib_lp_tests)
endif()
