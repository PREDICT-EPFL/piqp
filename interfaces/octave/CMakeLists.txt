# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

#  cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_OCTAVE_INTERFACE=ON .

# Note: Octave gets confused if there is a piqp.m and piqp.oct file in same directory. That's why
# we use the piqp_plain.oct output
set(TARGET_NAME piqp_plain.oct)

add_custom_target(${TARGET_NAME}
      COMMAND mkoctfile -v -o ${TARGET_NAME} -I${CMAKE_CURRENT_SOURCE_DIR}/../../include/piqp -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I/usr/include/eigen3 piqp_oct.cpp
      VERBATIM )

add_dependencies(${TARGET_NAME} piqp::piqp_c_shared)
