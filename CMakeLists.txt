# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5)

project(piqp
    VERSION 0.0.1
    LANGUAGES C CXX
)

#### User options ####
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

#### Developer options ####
option(ENABLE_SANITIZERS "Build with sanitizers enabled" OFF)

set(CMAKE_CXX_STANDARD 14)

# Set build type to RELEASE by default
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "" FORCE)
    endif()
endif()
message(STATUS "Building ${CMAKE_BUILD_TYPE}")

# Set compiler flags according to compiler ID
set(gcc_like_cxx CXX ARMClang AppleClang Clang GNU LCC)
set(msvc_like CXX MSVC)
if (${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
    list(APPEND compiler_flags -Wall -Wextra)
elseif ((${CMAKE_CXX_COMPILER_ID} IN_LIST msvc_like))
    list(APPEND compiler_flags -W3)
endif ()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Set sanitizer flags
if (ENABLE_SANITIZERS)
    list(APPEND sanitizer_flags -fsanitize=address -fno-omit-frame-pointer)
    list(APPEND compiler_flags ${sanitizer_flags})
    message(STATUS "Building with sanitizers: ${sanitizer_flags}")
    unset(sanitizer_flags)
endif ()

# Find Eigen3.3+
if (DEFINED EIGEN3_INCLUDE_DIRS)
    # Create target for user-defined Eigen 3.3+ path to be used for piqp
    add_library(Eigen3.3 INTERFACE IMPORTED)
    target_include_directories(Eigen3.3 INTERFACE ${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}") # Use custom Eigen path instead of package
else ()
    # Wrap Eigen3 3.3+ from system package into imported target for consistent use afterwards
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    add_library(Eigen3.3 INTERFACE IMPORTED)
    target_link_libraries(Eigen3.3 INTERFACE Eigen3::Eigen)
    message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}")
endif ()

add_library(piqp INTERFACE)
target_include_directories(piqp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(piqp INTERFACE Eigen3.3)
add_library(piqp::piqp ALIAS piqp)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS piqp
    EXPORT piqpTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html
install(
    EXPORT piqpTargets
    FILE piqpTargets.cmake
    DESTINATION lib/cmake/piqp
)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/piqpConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/piqpConfig.cmake
    INSTALL_DESTINATION lib/cmake/piqp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/piqpConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/piqpConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/piqpConfigVersion.cmake"
    DESTINATION lib/cmake/piqp
)

export(EXPORT piqpTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/piqpTargets.cmake
    NAMESPACE piqp::
)
