# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 OLD)
endif()

# Find pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# detect arm64 cross compilation on macOS
if(DEFINED ENV{_PYTHON_HOST_PLATFORM})
    if($ENV{_PYTHON_HOST_PLATFORM} MATCHES "arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

# add instruction set detection on x86/amd64
pybind11_add_module(instruction_set src/instruction_set.cpp)

# only link cpu_features on x86/amd64
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_definitions(instruction_set PRIVATE CPU_FEATURES_AVAILABLE=1)
    target_link_libraries(instruction_set PRIVATE cpu_features)
endif()

function(CREATE_PIQP_PYTHON_TARGET target_name COMPILE_OPTIONS)
    pybind11_add_module(${target_name} src/piqp_python.cpp)
    target_compile_options(${target_name} PRIVATE ${COMPILE_OPTIONS})
    target_link_libraries(${target_name} PRIVATE piqp_header_only_no_blasfeo_linked)
    target_compile_definitions(${target_name} PRIVATE PYTHON_MODULE_NAME=${target_name})
    # SKBUILD_PROJECT_VERSION_FULL is defined by scikit-build-core and passed into the C++ code as a define (VERSION_INFO) here.
    target_compile_definitions(${target_name} PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION_FULL})
    if(UNIX AND NOT APPLE)
        target_link_options(${target_name} PRIVATE "-Wl,-z,noexecstack")
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(AVX_COMPILE_OPTION "/arch:AVX")
    set(AVX2_COMPILE_OPTION "/arch:AVX2")
    set(FMA_COMPILE_OPTION "/fp:fast")
    set(AVX512_COMPILE_OPTION "/arch:AVX512")
else()
    set(AVX_COMPILE_OPTION "-mavx")
    set(AVX2_COMPILE_OPTION "-mavx2")
    set(FMA_COMPILE_OPTION "-mfma;-ffp-contract=on")
    set(AVX512_COMPILE_OPTION "-mavx512f")
endif()

create_piqp_python_target(piqp_python "")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    create_piqp_python_target(piqp_python_avx2 "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
    create_piqp_python_target(piqp_python_avx512 "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
endif()

if (BUILD_WITH_BLASFEO)
    # if x86/amd64
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        # if 64bit
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86_64)|(X86_64)|(amd64)|(AMD64)")
            if (TARGET blasfeo_x64)
                target_link_libraries(piqp_python PRIVATE blasfeo_x64)
            else ()
                target_link_libraries(piqp_python PRIVATE blasfeo)
            endif ()
            if (TARGET blasfeo_x64_avx2)
                target_link_libraries(piqp_python_avx2 PRIVATE blasfeo_x64_avx2)
            else ()
                target_link_libraries(piqp_python_avx2 PRIVATE blasfeo)
            endif ()
            if (TARGET blasfeo_x64_avx512)
                target_link_libraries(piqp_python_avx512 PRIVATE blasfeo_x64_avx512)
            else ()
                target_link_libraries(piqp_python_avx512 PRIVATE blasfeo)
            endif ()
        else ()
            target_link_libraries(piqp_python PRIVATE blasfeo)
            target_link_libraries(piqp_python_avx2 PRIVATE blasfeo)
            target_link_libraries(piqp_python_avx512 PRIVATE blasfeo)
        endif ()

    # if arm64
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(ARM64)|(arm64)|(aarch64)")
        if (TARGET blasfeo_arm64)
            target_link_libraries(piqp_python PRIVATE blasfeo_arm64)
        else ()
            target_link_libraries(piqp_python PRIVATE blasfeo)
        endif ()
    else ()
        target_link_libraries(piqp_python PRIVATE blasfeo)
    endif()
endif ()

install(TARGETS instruction_set LIBRARY DESTINATION piqp)
install(TARGETS piqp_python LIBRARY DESTINATION piqp)
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    install(TARGETS piqp_python_avx2 LIBRARY DESTINATION piqp)
    install(TARGETS piqp_python_avx512 LIBRARY DESTINATION piqp)
endif()
