# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

add_executable(cpp_dense_example cpp_dense_example.cpp)
target_link_libraries(cpp_dense_example PRIVATE piqp::piqp)

add_executable(cpp_sparse_example cpp_sparse_example.cpp)
target_link_libraries(cpp_sparse_example PRIVATE piqp::piqp)
