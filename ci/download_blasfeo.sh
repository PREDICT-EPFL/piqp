#!/bin/bash

# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

echo "Downloading blasfeo..."

# pinned, upstream 0a3df868 breaks the armv8a build (blasfeo_target.h moved)
git clone https://github.com/giaf/blasfeo.git blasfeo
git -C blasfeo checkout 6b182bd8c6b5358a8eadc0604dc7b830d39028b1
