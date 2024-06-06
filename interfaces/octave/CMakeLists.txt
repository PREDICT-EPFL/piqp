# This file is part of PIQP.
#
# Copyright (c) 2024 Joshua Redstone
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

# Note: Octave gets confused if there is a piqp.m and piqp.oct file in same directory. That's why
# we use the piqp_plain.oct output
set(TARGET_NAME piqp_plain.oct)

# mkoctfile is a utility program included with Octave
# TODO - piqp_plain_oct will be rebuilt every time. Add proper cmake smarts so it is only rebuilt when necessary
add_custom_target(${TARGET_NAME}
      COMMAND mkoctfile -v -o ${TARGET_NAME} -I${CMAKE_CURRENT_SOURCE_DIR}/../../include/piqp -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I/usr/include/eigen3 piqp_oct.cpp
      VERBATIM )

add_dependencies(${TARGET_NAME} piqp::piqp)
