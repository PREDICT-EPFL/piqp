# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

project(piqp
    VERSION 0.3.1
    LANGUAGES C CXX
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

#### Interface options ####
option(BUILD_WITH_TEMPLATE_INSTANTIATION "Build a shared library with common templates instantiated" ON)
option(BUILD_SHARED_LIBS "Build library as shared." ON)
option(BUILD_WITH_STD_OPTIONAL "Build using std::optional" OFF)
option(BUILD_WITH_STD_FILESYSTEM "Build using std::filesystem" OFF)
option(BUILD_C_INTERFACE "Build C interface" ON)
option(BUILD_PYTHON_INTERFACE "Build Python interface" OFF)
option(BUILD_MATLAB_INTERFACE "Build Matlab interface" OFF)
option(BUILD_OCTAVE_INTERFACE "Build Octave interface" OFF)

#### Tests/Benchmarks options ####
# Don't build tests and examples if included as subdirectory
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(BUILD_TESTS "Build tests" OFF)
    option(BUILD_EXAMPLES "Build examples" OFF)
else()
    option(BUILD_TESTS "Build tests" ON)
    option(BUILD_EXAMPLES "Build examples" ON)
endif()
option(BUILD_MAROS_MESZAROS_TEST "Build maros meszaros tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

#### Developer options ####
option(ENABLE_SANITIZERS "Build with sanitizers enabled" OFF)
option(DEBUG_PRINTS "Print additional debug information" OFF)

#### Install options ####
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(ENABLE_INSTALL "Enable install targets" OFF)
else()
    option(ENABLE_INSTALL "Enable install targets" ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if (BUILD_WITH_STD_OPTIONAL OR BUILD_WITH_STD_FILESYSTEM)
    set(CMAKE_CXX_STANDARD 17)
else ()
    set(CMAKE_CXX_STANDARD 14)
endif ()

# Set build type to RELEASE by default
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()
message(STATUS "Building ${CMAKE_BUILD_TYPE}")

# Set compiler flags according to compiler ID
set(gcc_like_cxx CXX ARMClang AppleClang Clang GNU LCC)
set(msvc_like CXX MSVC)
if (${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
    list(APPEND compiler_flags -Wall -Wextra -Wconversion -pedantic)
elseif ((${CMAKE_CXX_COMPILER_ID} IN_LIST msvc_like))
    list(APPEND compiler_flags /W4)
endif ()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Set sanitizer flags
if (ENABLE_SANITIZERS)
    if (${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
        list(APPEND sanitizer_flags -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    elseif ((${CMAKE_CXX_COMPILER_ID} IN_LIST msvc_like))
        list(APPEND sanitizer_flags /fsanitize=address)
    endif ()
    list(APPEND compiler_flags ${sanitizer_flags})
    message(STATUS "Building with sanitizers: ${sanitizer_flags}")
    unset(sanitizer_flags)
endif ()

if (BUILD_PYTHON_INTERFACE OR BUILD_MATLAB_INTERFACE)
    # building for conda-forge, TARGET_OS_OSX is not properly set, i.e., macOS is not correctly detected
    if (DEFINED ENV{CONDA_TOOLCHAIN_BUILD} AND APPLE)
        add_definitions(-DTARGET_OS_OSX=1)
    endif ()

    # Find cpu_features
    include(FetchContent)
    FetchContent_Declare(
        cpu_features
        URL https://github.com/google/cpu_features/archive/refs/tags/v0.9.0.zip
    )
    set(BUILD_SHARED_LIBS_COPY ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_GetProperties(cpu_features)
    if(NOT cpu_features_POPULATED)
        FetchContent_Populate(cpu_features)
        add_subdirectory(${cpu_features_SOURCE_DIR} ${cpu_features_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_COPY})
endif ()

if (BUILD_TESTS OR BUILD_BENCHMARKS)
    # Find Matio
    find_package(Matio REQUIRED)
endif ()

# Find Eigen3.3+
if (DEFINED EIGEN3_INCLUDE_DIRS AND NOT TARGET Eigen3::Eigen)
    # Create target for user-defined Eigen 3.3+ path to be used for piqp
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIRS})
else ()
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif ()
message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}")

macro(create_piqp_library library_name)
    set(options INTERFACE)
    cmake_parse_arguments(CREATE_PIQP_LIBRARY "${options}" "" "" ${ARGN})

    if (CREATE_PIQP_LIBRARY_INTERFACE)
        add_library(${library_name} INTERFACE)
        target_include_directories(${library_name} INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
            $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
        )
        target_link_libraries(${library_name} INTERFACE Eigen3::Eigen)
    else ()
        add_library(${library_name} ${TEMPLATE_SOURCES})
        target_include_directories(${library_name} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
            $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
        )
        target_link_libraries(${library_name} PUBLIC Eigen3::Eigen)
    endif ()
endmacro()

if (BUILD_WITH_TEMPLATE_INSTANTIATION)
    message(STATUS "Building with template instantiation")

    file(GLOB_RECURSE TEMPLATE_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

    create_piqp_library(piqp)
    set_target_properties(piqp PROPERTIES OUTPUT_NAME piqp)
    target_compile_definitions(piqp PUBLIC PIQP_WITH_TEMPLATE_INSTANTIATION)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        # Ensure ABI compatibility up to AVX512
        target_compile_definitions(piqp PUBLIC EIGEN_MAX_ALIGN_BYTES=64)
    endif ()
    target_compile_options(piqp PRIVATE ${compiler_flags})
    target_link_options(piqp PRIVATE ${compiler_flags})

    create_piqp_library(piqp_header_only INTERFACE)
else ()
    message(STATUS "Building without template instantiation")

    create_piqp_library(piqp INTERFACE)
    create_piqp_library(piqp_header_only INTERFACE)
endif ()

if (BUILD_WITH_STD_OPTIONAL)
    target_compile_definitions(piqp INTERFACE PIQP_STD_OPTIONAL)
    target_compile_definitions(piqp_header_only INTERFACE PIQP_STD_OPTIONAL)
endif ()
if (BUILD_WITH_STD_FILESYSTEM)
    target_compile_definitions(piqp INTERFACE PIQP_STD_FILESYSTEM)
    target_compile_definitions(piqp_header_only INTERFACE PIQP_STD_FILESYSTEM)
endif ()

if (DEBUG_PRINTS)
    target_compile_definitions(piqp INTERFACE PIQP_DEBUG_PRINT)
    target_compile_definitions(piqp_header_only INTERFACE PIQP_DEBUG_PRINT)
endif ()

add_library(piqp::piqp ALIAS piqp)
add_library(piqp::piqp_header_only ALIAS piqp_header_only)

macro(fix_test_dll target)
    if (WIN32)
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS
        )
    endif ()
endmacro()

if (BUILD_C_INTERFACE)
    add_subdirectory(interfaces/c)
endif()

if (BUILD_PYTHON_INTERFACE)
    add_subdirectory(interfaces/python)
endif()

if (BUILD_MATLAB_INTERFACE)
    add_subdirectory(interfaces/matlab)
endif()

if (BUILD_OCTAVE_INTERFACE)
    add_subdirectory(interfaces/octave)
endif()

if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (ENABLE_INSTALL)
    include(GNUInstallDirs)

    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        TARGETS piqp piqp_header_only
        EXPORT piqpTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html
    install(
        EXPORT piqpTargets
        FILE piqpTargets.cmake
        NAMESPACE piqp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/piqp
    )

    include(CMakePackageConfigHelpers)
    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/piqpConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/piqpConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/piqp
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/piqpConfigVersion.cmake
        VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/piqpConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/piqpConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/piqp
    )

    export(EXPORT piqpTargets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/piqpTargets.cmake
        NAMESPACE piqp::
    )

endif ()
