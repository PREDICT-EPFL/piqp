% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

function pre_install (in)
    current_dir = pwd;
    octave_interface_dir = fullfile(pwd, 'src/piqp/interfaces/octave');
    addpath(octave_interface_dir);

    setenv('PIQP_EIGEN3_INCLUDE_DIRS', fullfile(pwd, 'src/eigen'));
    make_piqp();

    files = {
        'piqp_instruction_set_oct.oct', ...
        'piqp_oct.oct', ...
        'piqp_avx2_oct.oct', ...
        'piqp_avx512_oct.oct'};
    for i = 1:length(files)
        file = files{i};
        if exist(fullfile(octave_interface_dir, file), 'file')
            copyfile(fullfile(octave_interface_dir, file), ...
                fullfile(fullfile(pwd, 'inst'), file));
        end
    end

    rmpath(octave_interface_dir)
endfunction
