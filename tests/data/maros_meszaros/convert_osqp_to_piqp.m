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

    x_lb = l(end-n+1:end);
    x_ub = u(end-n+1:end);
    
    C = A(1:end-n,:);
    cl = l(1:end-n);
    cu = u(1:end-n);

    eq_bounds = cu == cl;
    ineq_bounds = ~eq_bounds;

    A = C(eq_bounds, :);
    b = cu(eq_bounds);

    G = [C(ineq_bounds, :); -C(ineq_bounds, :)];
    h = [cu(ineq_bounds); -cl(ineq_bounds)];

    G = G(h < inf, :);
    h = h(h < inf);

    save(file.name, "P", "c", "A", "b", "G", "h", "x_lb", "x_ub");
end
