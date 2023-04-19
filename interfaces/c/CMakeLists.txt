# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5)

add_library(piqp_c_static STATIC src/piqp.cpp include/piqp.h)
target_include_directories(piqp_c_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(piqp_c_static PRIVATE piqp)
set_target_properties(piqp_c_static PROPERTIES OUTPUT_NAME piqp)
add_library(piqp::piqp_c_static ALIAS piqp_c_static)

add_library(piqp_c_shared SHARED src/piqp.cpp include/piqp.h)
target_include_directories(piqp_c_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(piqp_c_shared PRIVATE piqp)
set_target_properties(piqp_c_shared PROPERTIES OUTPUT_NAME piqp)
add_library(piqp::piqp_c_shared ALIAS piqp_c_shared)

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS piqp_c_static
    EXPORT piqpTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(
    TARGETS piqp_c_shared
    EXPORT piqpTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()