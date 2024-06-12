% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

confirm_recursive_rmdir(false);

%% Setup directories
current_dir = pwd;
[piqp_octave_dir,~,~] = fileparts(which('package_piqp.m'));
piqp_dir = fullfile(piqp_octave_dir, '../..');
tmp_dir = tempdir;

fprintf('Packaging PIQP...');

% Setup directory and copy files
pkg_name = 'piqp-octave';
pkg_dir = fullfile(tmp_dir, pkg_name);
if exist(pkg_dir, 'dir')
    rmdir(pkg_dir, 's');
end

% Copy files
copyfile(fullfile(piqp_octave_dir, 'package'), pkg_dir);
mkdir(fullfile(pkg_dir, 'inst'))
copyfile(fullfile(piqp_octave_dir, 'piqp.m'), fullfile(pkg_dir, 'inst/piqp.m'));

mkdir(fullfile(pkg_dir, 'src'));
copyfile(piqp_dir, fullfile(pkg_dir, 'src/piqp'));
rmdir(fullfile(pkg_dir, 'src/piqp/.git'), 's');

fprintf('Downloading Eigen3...');
system('wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz');
untar('eigen-3.4.0.tar.gz', fullfile(pkg_dir, 'src'));
delete('eigen-3.4.0.tar.gz');
movefile(fullfile(pkg_dir, 'src/eigen-3.4.0'), fullfile(pkg_dir, 'src/eigen'));

% Create tarball
cd(tmp_dir);
pkg_tar = sprintf('%s.tar.gz', pkg_name)
tar(pkg_tar, pkg_name);
rmdir(pkg_dir, 's');
movefile(pkg_tar, fullfile(piqp_octave_dir, pkg_tar));

cd(current_dir);
