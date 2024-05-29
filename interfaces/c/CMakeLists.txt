# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

add_library(piqp_c src/piqp.cpp include/piqp.h)
target_include_directories(piqp_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}/piqp_c>
)
target_link_libraries(piqp_c PRIVATE piqp_header_only)
set_target_properties(piqp_c PROPERTIES OUTPUT_NAME piqpc)
add_library(piqp::piqp_c ALIAS piqp_c)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/piqp_c
)

install(
    TARGETS piqp_c
    EXPORT piqpTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()
