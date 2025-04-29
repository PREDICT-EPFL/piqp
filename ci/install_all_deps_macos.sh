#!/bin/bash

# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p build_external
cd build_external

bash "$SCRIPT_DIR/install_eigen.sh"

bash "$SCRIPT_DIR/download_blasfeo.sh"
bash "$SCRIPT_DIR/build_install_blasfeo.sh" GENERIC generic

case $(uname -m) in
    x86_64|AMD64)
        bash "$SCRIPT_DIR/build_install_blasfeo.sh" X64_INTEL_CORE x64 /opt/blasfeo_x64
        bash "$SCRIPT_DIR/build_install_blasfeo.sh" X64_INTEL_HASWELL x64_avx2 /opt/blasfeo_x64_avx2
        bash "$SCRIPT_DIR/build_install_blasfeo.sh" X64_INTEL_SKYLAKE_X x64_avx512 /opt/blasfeo_x64_avx512
        ;;
    aarch64|arm64|ARM64)
        bash "$SCRIPT_DIR/build_install_blasfeo.sh" ARMV8A_APPLE_M1 arm64 /opt/blasfeo_arm64
        ;;
esac
