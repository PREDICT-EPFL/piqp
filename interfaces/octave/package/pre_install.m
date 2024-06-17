% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

function pre_install (in)
    octave_interface_dir = fullfile(pwd, 'src/piqp/interfaces/octave');
    addpath(octave_interface_dir);

    make_piqp();
    copyfile(fullfile(octave_interface_dir, 'piqp_oct.oct'), fullfile(pwd, 'inst/piqp_oct.oct'));

    rmpath(octave_interface_dir)
endfunction
