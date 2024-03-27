% This file is part of PIQP.
%
% Copyright (c) 2023 EPFL
% Copyright (c) 2021 Bartolomeo Stellato
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

function install_piqp
    % Install the PIQP solver Matlab interface

    % Get current operating system
    if ispc
        platform = 'windows';
    elseif ismac
        if strcmp(mexext, 'mexmaci64')
            platform = 'maci';
        else
            platform = 'maca';
        end
    elseif isunix
        platform = 'linux';
    end

    fprintf('Downloading binaries...');
    package_name = sprintf('https://github.com/PREDICT-EPFL/piqp/releases/latest/download/piqp-matlab-%s64.tar.gz', platform);
    websave('piqp.tar.gz', package_name);
    fprintf('\t\t\t\t[done]\n');

    fprintf('Unpacking...');
    untar('piqp.tar.gz','piqp')
    fprintf('\t\t\t\t\t[done]\n');

    fprintf('Updating path...');
    cd piqp
    addpath(genpath(pwd));
    savepath
    cd ..
    fprintf('\t\t\t\t[done]\n');

    fprintf('Deleting temporary files...');
    delete('piqp.tar.gz');
    fprintf('\t\t\t[done]\n');

    fprintf('PIQP is successfully installed!\n');


end