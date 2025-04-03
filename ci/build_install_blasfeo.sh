#!/bin/bash

# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

BLASFEO_TARGET=${1:-GENERIC};
ARCH_SUFFIX=${2:-};

echo "Installing blasfeo..."

cd blasfeo
mkdir "build_$ARCH_SUFFIX"
cd "build_$ARCH_SUFFIX"
case "$(uname -sr)" in
  CYGWIN*|MINGW*|MINGW32*|MSYS*) # detect windows
    cmake .. -G"Ninja" -DCMAKE_C_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBLASFEO_CROSSCOMPILING=ON -DTARGET="$BLASFEO_TARGET" -DBLAS_API=OFF -DBLASFEO_EXAMPLES=OFF ${3:+-DCMAKE_INSTALL_PREFIX="$3"}
    cmake --build .
    cmake --install .
    ;;
  *) # other OS
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBLASFEO_CROSSCOMPILING=ON -DTARGET="$BLASFEO_TARGET" -DBLAS_API=OFF -DBLASFEO_EXAMPLES=OFF ${3:+-DCMAKE_INSTALL_PREFIX="$3"}
    cmake --build .
    if [ "$EUID" -ne 0 ] # check if already root
      then
        sudo cmake --install .
      else
        cmake --install .
    fi
    ;;
esac
