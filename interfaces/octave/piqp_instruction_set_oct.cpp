// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <octave/oct.h>

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

DEFUN_DLD(piqp_instruction_set_oct, args, , "")
{
    // unused
    (void) args;

    octave_scalar_map info_struct;
#if defined(CPU_FEATURES_ARCH_X86)
    info_struct.assign("avx2", octave_value((bool) info.features.avx2));
    info_struct.assign("avx512f", octave_value((bool) info.features.avx512f));
#else
    info_struct.assign("avx2", octave_value(false));
    info_struct.assign("avx512f", octave_value(false));
#endif
    return octave_value(info_struct);
}
