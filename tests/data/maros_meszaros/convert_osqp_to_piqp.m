% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

files = dir(fullfile('../osqp_maros_meszaros_data/*.mat'));

for file = files'
    load(strcat(file.folder, "/", file.name));

    % P = P;
    c = q;

    l(l > 9e19) = inf;
    u(u > 9e19) = inf;
    l(l < -9e19) = -inf;
    u(u < -9e19) = -inf;

    x_l = l(end-n+1:end);
    x_u = u(end-n+1:end);
    
    C = A(1:end-n,:);
    cl = l(1:end-n);
    cu = u(1:end-n);

    eq_bounds = cu == cl;
    ineq_bounds = ~eq_bounds;

    A = C(eq_bounds, :)'';
    b = cu(eq_bounds);
    if numel(b) == 0, b = zeros(0, 1); end

    G = C(ineq_bounds, :);
    h_u = cu(ineq_bounds);
    h_l = cl(ineq_bounds);

    finite_bounds = (h_l > -inf) | (h_u < inf);
    G = G(finite_bounds, :)'';
    h_u = h_u(finite_bounds);
    h_l = h_l(finite_bounds);
    if numel(h_u) == 0, h_u = zeros(0, 1); end
    if numel(h_l) == 0, h_l = zeros(0, 1); end

    save(file.name, "P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u");
end
