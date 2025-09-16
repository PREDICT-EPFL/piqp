# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

find_package(Matlab)

# only link cpu_features on x86/amd64
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    matlab_add_mex(NAME piqp_instruction_set_mex SRC piqp_instruction_set_mex.cpp LINK_TO cpu_features)
    target_compile_definitions(piqp_instruction_set_mex PRIVATE CPU_FEATURES_AVAILABLE=1)
else ()
    matlab_add_mex(NAME piqp_instruction_set_mex SRC piqp_instruction_set_mex.cpp)
endif ()
set_target_properties(piqp_instruction_set_mex PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(piqp_instruction_set_mex PRIVATE ${compiler_flags})

function(CREATE_PIQP_MATLAB_TARGET target_name COMPILE_OPTIONS libs)
    matlab_add_mex(NAME ${target_name} SRC piqp_mex.cpp LINK_TO ${libs})
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

set(DEFAULT_LIBRARIES "piqp_header_only_no_blasfeo_linked")
set(AVX2_LIBRARIES "piqp_header_only_no_blasfeo_linked")
set(AVX512_LIBRARIES "piqp_header_only_no_blasfeo_linked")

if (BUILD_WITH_BLASFEO)
    # if x86/amd64
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        # if 64bit
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86_64)|(X86_64)|(amd64)|(AMD64)")
            if (TARGET blasfeo_x64)
                set(DEFAULT_LIBRARIES "${DEFAULT_LIBRARIES};blasfeo_x64")
            else ()
                set(DEFAULT_LIBRARIES "${DEFAULT_LIBRARIES};blasfeo")
            endif ()
            if (TARGET blasfeo_x64_avx2)
                set(AVX2_LIBRARIES "${AVX2_LIBRARIES};blasfeo_x64_avx2")
            else ()
                set(AVX2_LIBRARIES "${AVX2_LIBRARIES};blasfeo")
            endif ()
            if (TARGET blasfeo_x64_avx512)
                set(AVX512_LIBRARIES "${AVX512_LIBRARIES};blasfeo_x64_avx512")
            else ()
                set(AVX512_LIBRARIES "${AVX512_LIBRARIES};blasfeo")
            endif ()
        else ()
            set(DEFAULT_LIBRARIES "${DEFAULT_LIBRARIES};blasfeo")
            set(AVX2_LIBRARIES "${AVX2_LIBRARIES};blasfeo")
            set(AVX512_LIBRARIES "${AVX512_LIBRARIES};blasfeo")
        endif ()

        # if arm64
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(ARM64)|(arm64)|(aarch64)")
        if (TARGET blasfeo_arm64)
            set(DEFAULT_LIBRARIES "${DEFAULT_LIBRARIES};blasfeo_arm64")
        else ()
            set(DEFAULT_LIBRARIES "${DEFAULT_LIBRARIES};blasfeo")
        endif ()
    else ()
        set(DEFAULT_LIBRARIES "${DEFAULT_LIBRARIES};blasfeo")
    endif()
endif ()

create_piqp_matlab_target(piqp_mex "" "${DEFAULT_LIBRARIES}")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    create_piqp_matlab_target(piqp_avx2_mex "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}" "${AVX2_LIBRARIES}")
    create_piqp_matlab_target(piqp_avx512_mex "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}" "${AVX512_LIBRARIES}")
endif()
