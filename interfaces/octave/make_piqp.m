% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
% Copyright (c) 2018 Bartolomeo Stellato, Paul Goulart, Goran Banjac
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

function make_piqp(varargin)
% Octave OCT makefile for PIQP.
%
%    MAKE_PIQP(VARARGIN) is a make file for the PIQP solver. It
%    builds the PIQP Octave interface, i.e., oct files.
%
%    make_piqp            - build and package interface
%    make_piqp('oct')     - build oct files using CMake
%    make_piqp('clean')   - remove all build related files

confirm_recursive_rmdir(false);

if( nargin == 0 )
    what = {'all'};
    verbose = false;
elseif ( nargin == 1 && ismember('-verbose', varargin) )
    what = {'all'};
    verbose = true;
else
    what = varargin{nargin};
    if isempty(strfind(what, 'oct')) && ...
      isempty(strfind(what, 'clean'))
        fprintf('"%s" is not a valid command\n', what);
      return
    end

    verbose = ismember('-verbose', varargin);
end

%% Try to unlock any pre-existing version of piqp_oct
if mislocked('piqp_oct')
    munlock('piqp_oct');
end
if mislocked('piqp_avx2_oct')
    munlock('piqp_avx2_oct');
end
if mislocked('piqp_avx512_oct')
    munlock('piqp_avx512_oct');
end

%% Setup directories
current_dir = pwd;
[piqp_octave_dir,~,~] = fileparts(which('make_piqp.m'));
if exist(fullfile(piqp_octave_dir, 'src'))
    piqp_dir = fullfile(piqp_octave_dir, 'src/piqp');
else
    piqp_dir = fullfile(piqp_octave_dir, '../..');
end
piqp_build_dir = fullfile(piqp_dir, 'build');

%% Compile commands

% Get make and oct commands
make_cmd = 'cmake --build .';

% Add arguments to cmake and oct compiler
cmake_args = [
    '-DBUILD_C_INTERFACE=OFF ' ...
    '-DBUILD_OCTAVE_INTERFACE=ON ' ...
    '-DBUILD_TESTS=OFF ' ...
    '-DBUILD_EXAMPLES=OFF ' ...
    '-DBUILD_BENCHMARKS=OFF'];

if getenv('PIQP_EIGEN3_INCLUDE_DIRS')
    cmake_args = sprintf('%s -DEIGEN3_INCLUDE_DIRS=%s', cmake_args, getenv('PIQP_EIGEN3_INCLUDE_DIRS'));
end

%% piqp_oct
if any(strcmpi(what,'oct')) || any(strcmpi(what,'all'))
   fprintf('Compiling PIQP Octave interface...');

    % Create build directory and go inside
    if exist(piqp_build_dir, 'dir')
        rmdir(piqp_build_dir, 's');
    end
    mkdir(piqp_build_dir);
    cd(piqp_build_dir);

    % Extend path for CMake mac (via Homebrew)
    PATH = getenv('PATH');
    if (ismac && isempty(strfind(PATH, '/usr/local/bin')))
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


    % Change directory back to octave interface
    cd(piqp_octave_dir);

    fprintf('\t\t\t\t\t\t[done]\n');

end

%% clean
if any(strcmpi(what,'clean'))
    fprintf('Cleaning build related files...');

    % Change directory back to octave interface
    cd(piqp_octave_dir);

    % Delete oct file
    octfiles = dir('*.oct');
    for i = 1 : length(octfiles)
        delete(octfiles(i).name);
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