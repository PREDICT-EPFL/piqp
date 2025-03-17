% This file is part of PIQP.
%
% Copyright (c) 2023 EPFL
% Copyright (c) 2018 Bartolomeo Stellato, Paul Goulart, Goran Banjac
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

function make_piqp(varargin)
% Matlab MEX makefile for PIQP.
%
%    MAKE_PIQP(VARARGIN) is a make file for the PIQP solver. It
%    builds the PIQP Matlab interface, i.e., mex files, and
%    packages it.
%
%    make_piqp            - build and package interface
%    make_piqp('mex')     - build mex files using CMake 
%    make_piqp('package') - package interface
%    make_piqp('clean')   - remove all build related files

if( nargin == 0 )
    what = {'all'};
    verbose = false;
elseif ( nargin == 1 && ismember('-verbose', varargin) )
    what = {'all'};
    verbose = true;
else
    what = varargin{nargin};
    if ~contains(what, 'mex') && ...
      ~contains(what, 'package') && ...
      ~contains(what, 'clean') 
        fprintf('"%s" is not a valid command\n', what);
      return
    end

    verbose = ismember('-verbose', varargin);
end

%% Try to unlock any pre-existing version of piqp_mex
if mislocked('piqp_mex')
    munlock('piqp_mex');
end
if mislocked('piqp_avx2_mex')
    munlock('piqp_avx2_mex');
end
if mislocked('piqp_avx512_mex')
    munlock('piqp_avx512_mex');
end

%% Setup directories
current_dir = pwd;
[piqp_matlab_dir,~,~] = fileparts(which('make_piqp.m'));
piqp_dir = fullfile(piqp_matlab_dir, '../..');
piqp_build_dir = fullfile(piqp_dir, 'build');

%% Compile commands

% Get make and mex commands
make_cmd = 'cmake --build .';

% Add arguments to cmake and mex compiler
cmake_args = [
    '-DBUILD_C_INTERFACE=OFF ' ...
    '-DBUILD_WITH_TEMPLATE_INSTANTIATION=OFF ' ...
    '-DBUILD_MATLAB_INTERFACE=ON ' ...
    '-DBUILD_TESTS=OFF ' ...
    '-DBUILD_EXAMPLES=OFF ' ...
    '-DBUILD_BENCHMARKS=OFF ' ...
    getenv("CMAKE_ARGS")];

% Add specific generators for windows linux or mac
% if (ispc)
%     cmake_args = sprintf('%s %s', cmake_args, '-G "MinGW Makefiles"');
% else
%     cmake_args = sprintf('%s %s', cmake_args, '-G "Unix Makefiles"');
% end

% Pass Matlab root to cmake
Matlab_ROOT = strrep(matlabroot, '\', '/');
cmake_args = sprintf('%s %s%s%s', cmake_args, ...
    '-DMatlab_ROOT_DIR="', Matlab_ROOT, '"');

%% piqp_mex
if any(strcmpi(what,'mex')) || any(strcmpi(what,'all'))
   fprintf('Compiling PIQP Matlab interface...');

    % Create build directory and go inside
    if exist(piqp_build_dir, 'dir')
        rmdir(piqp_build_dir, 's');
    end
    mkdir(piqp_build_dir);
    cd(piqp_build_dir);

    % Extend path for CMake mac (via Homebrew)
    PATH = getenv('PATH');
    if (ismac && ~contains(PATH, '/usr/local/bin'))
        setenv('PATH', [PATH ':/usr/local/bin']);
    end

    % Compile static library with CMake
    [status, output] = system(sprintf('%s %s ..', 'cmake', cmake_args));
    if(status)
        fprintf('\n');
        disp(output);
        error('Error configuring CMake environment');
    elseif(verbose)
        fprintf('\n');
        disp(output);
    end

    [status, output] = system(make_cmd);
    if (status)
        fprintf('\n');
        disp(output);
        error('Error compiling PIQP Interface');
    elseif(verbose)
        fprintf('\n');
        disp(output);
    end


    % Change directory back to matlab interface
    cd(piqp_matlab_dir);

    fprintf('\t\t\t\t\t\t[done]\n');

end

%% Package  
if any(strcmpi(what,'package')) || any(strcmpi(what,'all'))
    fprintf('Packaging PIQP...');

    % Get platform
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
    
    % Setup directory and copy files
    pkg_name = sprintf('piqp-matlab-%s64', platform);
    if exist(fullfile(piqp_matlab_dir, pkg_name), 'dir')
        rmdir(pkg_name, 's');
    end
    mkdir(pkg_name);
    
    % Copy folders
    folders = {'tests'};
    for i = 1:length(folders)
        folder = folders{i};
        copyfile(fullfile(piqp_matlab_dir, folder), ...
            fullfile(pkg_name, folder));
    end

    % Copy files
    files = {
        'piqp.m', ...
        'piqp_instruction_set_mex.cpp', ...
        sprintf('piqp_instruction_set_mex.%s', mexext), ...
        'piqp_mex.cpp', ...
        sprintf('piqp_mex.%s', mexext), ...
        sprintf('piqp_avx2_mex.%s', mexext), ...
        sprintf('piqp_avx512_mex.%s', mexext), ...
        'runtest_piqp.m'};
    for i = 1:length(files)
        file = files{i};
        if exist(file, 'file')
            copyfile(fullfile(piqp_matlab_dir, file), ...
                fullfile(pkg_name, file));
        end
    end
    
    % Copy license
    copyfile(fullfile(piqp_dir, 'LICENSE'), fullfile(pkg_name));
    
    % Create tarball
    tar(sprintf('%s.tar.gz', pkg_name), pkg_name);
    rmdir(pkg_name, 's');

    fprintf('\t\t\t[done]\n');
end

%% clean
if any(strcmpi(what,'clean'))
    fprintf('Cleaning build related files...');

    % Change directory back to matlab interface
    cd(piqp_matlab_dir);

    % Delete mex file
    mexfiles = dir(['*.', mexext]);
    for i = 1 : length(mexfiles)
        delete(mexfiles(i).name);
    end

    % Delete PIQP build directory
    if exist(piqp_build_dir, 'dir')
        rmdir(piqp_build_dir, 's');
    end

    fprintf('\t\t\t[done]\n');
end

%% Go back to the original directory
cd(current_dir);

end