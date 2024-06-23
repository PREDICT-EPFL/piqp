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

fprintf('Packaging PIQP...\n');

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
copyfile(fullfile(piqp_octave_dir, 'make_piqp.m'), fullfile(pkg_dir, 'make_piqp.m'));

mkdir(fullfile(pkg_dir, 'src'));
copyfile(piqp_dir, fullfile(pkg_dir, 'src/piqp'));
rmdir(fullfile(pkg_dir, 'src/piqp/.git'), 's');
rmdir(fullfile(pkg_dir, 'src/piqp/benchmarks'), 's');
rmdir(fullfile(pkg_dir, 'src/piqp/tests'), 's');

fprintf('Downloading Eigen3...\n');
tar_path = fullfile(tmp_dir, 'eigen-3.4.0.tar.gz');
data = urlread('https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz');
if exist(tar_path, 'file')
    delete(tar_path);
end
tar_file = fopen(tar_path, 'w');
fwrite(tar_file, data);
fclose(tar_file);
untar(tar_path, fullfile(pkg_dir, 'src'));
delete(tar_path);
movefile(fullfile(pkg_dir, 'src/eigen-3.4.0'), fullfile(pkg_dir, 'src/eigen'));

cd(tmp_dir);
pkg_tar = sprintf('%s.tar', pkg_name);
pkg_tar_gz = sprintf('%s.tar.gz', pkg_name);
fprintf('Creating %s\n', pkg_tar_gz);
tar(pkg_tar, pkg_name);
gzip(pkg_tar);
rmdir(pkg_dir, 's');
delete(pkg_tar);
movefile(pkg_tar_gz, fullfile(piqp_octave_dir, pkg_tar_gz));

cd(current_dir);
