% This file is part of PIQP.
%
% Copyright (c) 2023 EPFL
% Copyright (c) 2019 Bartolomeo Stellato, Goran Banjac
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

import matlab.unittest.TestSuite;

[root_path,~,~] = fileparts(which('runtest_piqp.m'));
addpath(root_path)
tests_path = fullfile(root_path, 'tests');

suite = TestSuite.fromFolder(tests_path);
result = run(suite);

% Always throw error on failure - CI will catch it, GUI will show it
if any([result.Failed]) || any([result.Incomplete])
    error('Tests failed! %d failed, %d incomplete', ...
          sum([result.Failed]), sum([result.Incomplete]));
end
