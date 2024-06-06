# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

project(piqp_examples)

set(CMAKE_CXX_STANDARD 14)

# find_package(piqp REQUIRED)

if (BUILD_C_INTERFACE)
    add_subdirectory(c)
endif()
add_subdirectory(cpp)
