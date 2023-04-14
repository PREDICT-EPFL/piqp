# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/pybind11)
add_subdirectory(external/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/external/pybind11)

# add instruction set detection on x86/amd64
pybind11_add_module(instruction_set src/instruction_set.cpp)

# only link cpu_features on x86/amd64
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(BUILD_TESTING OFF)
    add_subdirectory(external/cpu_features)
    target_compile_definitions(instruction_set PRIVATE CPU_FEATURES_AVAILABLE=1)
    target_link_libraries(instruction_set PRIVATE CpuFeature::cpu_features)
endif()

function(CREATE_PIQP_PYTHON_TARGET target_name COMPILE_OPTIONS)
    pybind11_add_module(${target_name} src/piqp_python.cpp)
    target_compile_features(${target_name} PRIVATE cxx_std_17) # C++17 support needed for std::optional
    target_compile_options(${target_name} PRIVATE ${COMPILE_OPTIONS})
    target_link_libraries(${target_name} PRIVATE piqp)
    target_compile_definitions(${target_name} PRIVATE PYTHON_MODULE_NAME=${target_name})
    # PIQP_VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
    target_compile_definitions(${target_name} PRIVATE VERSION_INFO=${PIQP_VERSION_INFO})
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(AVX_COMPILE_OPTION "/arch:AVX")
    set(AVX2_COMPILE_OPTION "/arch:AVX2")
    set(FMA_COMPILE_OPTION "/fp:fast,")
    set(AVX512_COMPILE_OPTION "/arch:AVX512")
else()
    set(AVX_COMPILE_OPTION "-mavx")
    set(AVX2_COMPILE_OPTION "-mavx2")
    set(FMA_COMPILE_OPTION "-mfma")
    set(AVX512_COMPILE_OPTION "-mavx512f")
endif()

create_piqp_python_target(piqp_python "")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    create_piqp_python_target(piqp_python_avx2 "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
    create_piqp_python_target(piqp_python_avx512 "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
endif()
