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
bash "$SCRIPT_DIR/build_install_blasfeo.sh GENERIC generic"
bash "$SCRIPT_DIR/build_install_blasfeo.sh X64_INTEL_CORE x64 c:/opt/blasfeo_x64"
bash "$SCRIPT_DIR/build_install_blasfeo.sh X64_INTEL_HASWELL x64_avx2 c:/opt/blasfeo_x64_avx2"
bash "$SCRIPT_DIR/build_install_blasfeo.sh X64_INTEL_SKYLAKE_X arm64 c:/opt/blasfeo_x64_avx512"
