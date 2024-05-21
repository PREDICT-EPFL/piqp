# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

#  cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_OCTAVE_INTERFACE=ON .

cmake_minimum_required(VERSION 3.5)

function(CREATE_PIQP_OCTAVE_TARGET target_name COMPILE_OPTIONS)
    octave_add_oct(${target_name} SOURCES piqp_oct.cpp LINK_LIBRARIES piqp_header_only)
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${target_name} PRIVATE ${COMPILE_OPTIONS})
    target_compile_options(${target_name} PRIVATE ${compiler_flags})
    target_compile_definitions(${target_name} PRIVATE PIQP_VERSION=${CMAKE_PROJECT_VERSION})
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

create_piqp_octave_target(piqp_oct "")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    create_piqp_octave_target(piqp_avx2_oct "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
    create_piqp_octave_target(piqp_avx512_oct "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
endif()
