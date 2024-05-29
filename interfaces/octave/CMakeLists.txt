# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

#  cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_OCTAVE_INTERFACE=ON .

#cmake_minimum_required(VERSION 3.5)

#find_package(Octave)  # Calls FindOctave

function(CREATE_PIQP_OCTAVE_TARGET target_base COMPILE_OPTIONS)
  # -lpiqpcstatic -L${CMAKE_CURRENT_SOURCE_DIR}/../c
  # -I${CMAKE_CURRENT_SOURCE_DIR}/../c/include 
  # Note: Octave gets confused if there is a piqp.m and piqp.oct file in same directory. That's why
  # we use the piqp_plain.oct output

  set(TARGET_NAME "${target_base}.oct")
  add_custom_target(${TARGET_NAME}
                     COMMAND ${CMAKE_COMMAND} -E env CPPFLAGS="${COMPILE_OPTIONS}" -- mkoctfile -v -o ${TARGET_NAME} -I${CMAKE_CURRENT_SOURCE_DIR}/../../include/piqp -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I/usr/include/eigen3 piqp_oct.cpp
          # NOT setting VERBATIM here.  It does not appear possible to get correct quoting around
          # COMPILE_OPTIONS if VERBATIM is set.  Either you get no quotes, or escaped quotes.
                     )
  add_dependencies(${TARGET_NAME} piqp::piqp_c_shared)
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

# Creates targe piqp_oct
create_piqp_octave_target(piqp_plain "")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    # Creates targe piqp_avx2_oct and piqp_avx512_oct
    create_piqp_octave_target(piqp_avx2  "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
    create_piqp_octave_target(piqp_avx512  "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
endif()
