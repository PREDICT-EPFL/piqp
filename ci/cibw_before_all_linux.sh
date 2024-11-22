#!/bin/bash

# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

mkdir -p build_external
cd build_external

bash ../ci/install_eigen.sh

bash ../ci/download_blasfeo.sh
bash ../ci/build_install_blasfeo.sh GENERIC generic

case $(uname -m) in
    x86_64)
        bash ../ci/build_install_blasfeo.sh X64_INTEL_CORE x64 /opt/blasfeo_x64
        bash ../ci/build_install_blasfeo.sh X64_INTEL_HASWELL x64_avx2 /opt/blasfeo_x64_avx2
        bash ../ci/build_install_blasfeo.sh X64_INTEL_SKYLAKE_X arm64 /opt/blasfeo_x64_avx512
        export CMAKE_ARGS="-DBLASFEO_X64_DIR=/opt/blasfeo_x64 -DBLASFEO_X64_AVX2_DIR=/opt/blasfeo_x64_avx2 -DBLASFEO_X64_AVX512_DIR=/opt/blasfeo_x64_avx512"
        ;;
    aarch64|arm64)
        bash ../ci/build_install_blasfeo.sh ARMV8A_APPLE_M1 arm64 /opt/blasfeo_arm64
        export CMAKE_ARGS="-DBLASFEO_ARM64_DIR=/opt/blasfeo_arm64"
        ;;
esac
