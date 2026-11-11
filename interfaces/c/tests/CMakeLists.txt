# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Keep the explicit FetchContent populate/add_subdirectory flow for CMake 3.21
# support while still excluding third-party install rules.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
    cmake_policy(SET CMP0169 OLD)
endif()

# Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        URL_HASH SHA256=a9607c9215866bd425a725610c5e0f739eeb50887a57903df48891446ce6fb3c
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

enable_testing()

add_library(piqp-c-interface-test INTERFACE)
target_include_directories(piqp-c-interface-test INTERFACE include)
target_compile_options(piqp-c-interface-test INTERFACE ${compiler_flags})
if(${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
    target_compile_options(piqp-c-interface-test INTERFACE -Wno-gnu-zero-variadic-macro-arguments -Wno-c++17-attribute-extensions)
endif()
target_link_options(piqp-c-interface-test INTERFACE ${linker_flags})
target_link_libraries(piqp-c-interface-test INTERFACE piqp_c piqp_header_only gtest_main gtest)

add_executable(c_interface_test src/c_interface_test.cpp)
target_link_libraries(c_interface_test PRIVATE piqp-c-interface-test)

fix_test_dll(c_interface_test)

include(GoogleTest)
gtest_discover_tests(c_interface_test)
