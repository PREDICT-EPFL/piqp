#!/bin/bash

# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

EIGEN_VERSION=${EIGEN_VERSION:-3.4.0};

echo "Installing EIGEN..."

git clone https://gitlab.com/libeigen/eigen.git eigen
cd eigen
git checkout "$EIGEN_VERSION"

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2

case "$(uname -sr)" in
  CYGWIN*|MINGW*|MINGW32*|MSYS*) # detect windows
    make install
    ;;
  *) # other OS
    if [ "$EUID" -ne 0 ] # check if already root
      then
        sudo make install
      else
        make install
    fi
    ;;
esac
