// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "mex.h"

#if defined(CPU_FEATURES_AVAILABLE)
#include "cpu_features_macros.h"
#endif

#if defined(CPU_FEATURES_ARCH_X86)
#include "cpuinfo_x86.h"
#endif

#if defined(CPU_FEATURES_AVAILABLE)
using namespace cpu_features;
#endif

#if defined(CPU_FEATURES_ARCH_X86)
const X86Info info = GetX86Info();
#endif

const char* INFO_FIELDS[] = { "avx2", "avx512f" };

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // unused
    (void) nlhs;
    (void) nrhs;
    (void) prhs;

    int n_info = sizeof(INFO_FIELDS) / sizeof(INFO_FIELDS[0]);
    mxArray* info_struct = mxCreateStructMatrix(1, 1, n_info, INFO_FIELDS);
#if defined(CPU_FEATURES_ARCH_X86)
    mxSetField(info_struct, 0, "avx2", mxCreateLogicalScalar((bool) info.features.avx2));
    mxSetField(info_struct, 0, "avx512f", mxCreateLogicalScalar((bool) info.features.avx512f));
#else
    mxSetField(info_struct, 0, "avx2", mxCreateLogicalScalar(false));
    mxSetField(info_struct, 0, "avx512f", mxCreateLogicalScalar(false));
#endif
    plhs[0] = info_struct;
}
