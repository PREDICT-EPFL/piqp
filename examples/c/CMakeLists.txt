# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

add_executable(c_dense_example c_dense_example.c)
target_link_libraries(c_dense_example PRIVATE piqp::piqp_c)

add_executable(c_sparse_example c_sparse_example.c)
target_link_libraries(c_sparse_example PRIVATE piqp::piqp_c)
