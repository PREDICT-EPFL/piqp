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

%% Setup directories
current_dir = pwd;
[piqp_octave_dir,~,~] = fileparts(which('make_piqp.m'));
if exist(fullfile(piqp_octave_dir, 'src'))
    piqp_dir = fullfile(piqp_octave_dir, 'src/piqp');
    eigen_include_dir = fullfile(piqp_octave_dir, 'src/eigen');
else
    piqp_dir = fullfile(piqp_octave_dir, '../..');
    eigen_include_dir = '/usr/local/include/eigen3';
end

%% piqp_oct
if any(strcmpi(what,'oct')) || any(strcmpi(what,'all'))
    fprintf('Compiling PIQP Octave interface...\n');

    mkoctfile_args = {'-O3', '-DNDEBUG', '-march=native', ...
             ['-I', fullfile(piqp_dir, 'include')], ...
             ['-I', eigen_include_dir], ...
             '-o', 'piqp_oct.oct', ...
             fullfile(piqp_dir, 'interfaces/octave/piqp_oct.cpp')};

    if ~exist('verLessThan') || verLessThan("Octave", "10")
        % Octave 10 or newer requires C++17 or newer.
        % Attempting to lower that requirement to C++14 (with GNU extensions)
        % leads to compilation errors.
        mkoctfile_args = [{'-std=gnu++14'}, mkoctfile_args];
    end

    mkoctfile(mkoctfile_args{:});

    fprintf('[done]\n');

end

%% clean
if any(strcmpi(what,'clean'))
    fprintf('Cleaning build related files...\n');

    % Delete oct file
    octfiles = dir('*.oct');
    for i = 1 : length(octfiles)
        delete(octfiles(i).name);
    end

    fprintf('[done]\n');
end

%% Go back to the original directory
cd(current_dir);

end