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
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make -j2
sudo make install
