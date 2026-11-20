% This file is part of PIQP.
%
% Copyright (c) 2023 EPFL
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

classdef simple_tests < matlab.unittest.TestCase

    properties
        P
        c
        A
        b
        G
        h_l
        h_u
        x_l
        x_u
        solver_dense
        solver_sparse
        tol
    end

    methods(TestMethodSetup)
        function setup_problem(testCase)

            % Create Problem
            testCase.P = sparse([6 0; 0 4]);
            testCase.c = [-1; -4];
            testCase.A = sparse([1 -2]);
            testCase.b = 0;
            testCase.G = sparse([1 0; -1 0]);
            testCase.h_l = [-Inf; -Inf];
            testCase.h_u = [1; 1];
            testCase.x_l = [-Inf; -1];
            testCase.x_u = [Inf; 1];

            % Setup dense solver
            testCase.solver_dense = piqp('dense');
            testCase.solver_dense.setup( ...
                testCase.P, testCase.c, ...
                testCase.A, testCase.b, ...
                testCase.G, testCase.h_l, testCase.h_u, ...
                testCase.x_l, testCase.x_u, ...
                'verbose', true);
            testCase.solver_dense.update_settings('verbose', false);

            % Setup sparse solver
            testCase.solver_sparse = piqp('sparse');
            testCase.solver_sparse.setup( ...
                testCase.P, testCase.c, ...
                testCase.A, testCase.b, ...
                testCase.G, testCase.h_l, testCase.h_u, ...
                testCase.x_l, testCase.x_u, ...
                'verbose', false);
            testCase.solver_sparse.update_settings('verbose', false);

            % Tolerance for checking solutions
            testCase.tol = 1e-06;

        end
    end

    methods (Test)
        function solve_dense_qp(testCase)
            results = testCase.solver_dense.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.4285714; 0.2142857], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.y, -1.5714286, 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_l, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_u, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bl, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bu, [0; 0], 'AbsTol', testCase.tol)
        end
        
        function solve_sparse_qp(testCase)
            results = testCase.solver_sparse.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.4285714; 0.2142857], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.y, -1.5714286, 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_l, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_u, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bl, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bu, [0; 0], 'AbsTol', testCase.tol)
        end

        function update_dense_qp(testCase)
            testCase.solver_dense.solve();

            P_new = [8 0; 0 4];
            A_new = [1 -3];
            h_u_new = [2; 1];
            x_u_new = [Inf; 2];

            testCase.solver_dense.update('P', P_new, 'A', A_new, 'h_u', h_u_new, 'x_u', x_u_new);
            results = testCase.solver_dense.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.2763157; 0.0921056], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.y, -1.2105263, 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_l, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_u, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bl, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bu, [0; 0], 'AbsTol', testCase.tol)
        end

        function update_sparse_qp(testCase)
            testCase.solver_sparse.solve();

            P_new = sparse([8 0; 0 4]);
            A_new = sparse([1 -3]);
            h_u_new = [2; 1];
            x_u_new = [Inf; 2];

            testCase.solver_sparse.update('P', P_new, 'A', A_new, 'h_u', h_u_new, 'x_u', x_u_new);
            results = testCase.solver_sparse.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.2763157; 0.0921056], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.y, -1.2105263, 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_l, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_u, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bl, [0; 0], 'AbsTol', testCase.tol)
            testCase.verifyEqual(results.z_bu, [0; 0], 'AbsTol', testCase.tol)
        end

        function non_double_inputs(testCase)
            % single, integer and logical data is converted to double (all values are exactly representable)
            for backend = {'dense', 'sparse'}
                solver = piqp(backend{1});
                solver.setup(single(full(testCase.P)), int32(testCase.c), ...
                             int8(full(testCase.A)), logical(testCase.b), ...
                             int8(full(testCase.G)), testCase.h_l, single(testCase.h_u), ...
                             testCase.x_l, single(testCase.x_u));
                results = solver.solve();
                testCase.verifyEqual(results.x, [0.4285714; 0.2142857], 'AbsTol', testCase.tol)

                solver.update('c', single(testCase.c), 'P', int16(full(testCase.P)));
                results = solver.solve();
                testCase.verifyEqual(results.x, [0.4285714; 0.2142857], 'AbsTol', testCase.tol)

                % complex and non-numeric data is rejected
                testCase.verifyError(@() solver.update('c', complex(testCase.c)), 'piqp:input')
                testCase.verifyError(@() solver.update('h_u', 'ab'), 'piqp:input')
                other = piqp(backend{1});
                testCase.verifyError(@() other.setup(testCase.P, 1i * testCase.c, [], [], [], [], [], [], []), 'piqp:input')
            end
        end

        function vjp_dense_qp(testCase)
            check_vjp(testCase, 'dense');
        end

        function vjp_sparse_qp(testCase)
            check_vjp(testCase, 'sparse');
        end

        function vjp_not_enabled(testCase)
            testCase.solver_dense.solve();
            grads = testCase.solver_dense.vjp('grad_x', [1; 2]);
            testCase.verifyEqual(grads.status_val, -10) % PIQP_INVALID_SETTINGS
            testCase.verifyEmpty(grads.c)
        end

        function jvp_dense_qp(testCase)
            check_jvp(testCase, 'dense');
        end

        function jvp_sparse_qp(testCase)
            check_jvp(testCase, 'sparse');
        end

        function jvp_not_enabled(testCase)
            testCase.solver_dense.solve();
            tangents = testCase.solver_dense.jvp('dc', [1; 2]);
            testCase.verifyEqual(tangents.status_val, -10) % PIQP_INVALID_SETTINGS
            testCase.verifyEmpty(tangents.x)
        end
    end

    methods
        function solver = setup_vjp_solver(testCase, backend, c)
            % same problem with an active upper bound x2 <= 0.1
            solver = piqp(backend);
            solver.setup(testCase.P, c, testCase.A, testCase.b, ...
                         testCase.G, testCase.h_l, testCase.h_u, ...
                         testCase.x_l, [Inf; 0.1], ...
                         'eps_abs', 1e-12, 'eps_rel', 1e-12, 'enable_derivatives', true);
        end

        function check_vjp(testCase, backend)
            % loss L = x1 + 2 x2 + y
            solver = setup_vjp_solver(testCase, backend, testCase.c);
            grads = solver.vjp('grad_x', [1; 2], 'grad_y', 1);
            testCase.verifyEqual(grads.status_val, -9) % PIQP_UNSOLVED (no solution)
            testCase.verifyEmpty(grads.c)

            res = solver.solve();
            grads = solver.vjp('grad_x', [1; 2], 'grad_y', 1);
            testCase.verifyEqual(grads.status_val, 1) % PIQP_SOLVED

            % finite differences w.r.t. c
            eps = 1e-6;
            for i = 1:2
                dc = zeros(2, 1);
                dc(i) = eps;
                fd = (vjp_loss(testCase, backend, testCase.c + dc) - ...
                      vjp_loss(testCase, backend, testCase.c - dc)) / (2 * eps);
                testCase.verifyEqual(grads.c(i), fd, 'AbsTol', 1e-5)
            end
            testCase.verifyEqual(size(grads.P), [2 2])
            testCase.verifyEqual(size(grads.A), [1 2])
            testCase.verifyEqual(size(grads.G), [2 2])
            testCase.verifyEqual(issparse(grads.A), strcmp(backend, 'sparse'))

            % gradients at a provided solve result, the default call linearizes at the same point
            grads_fr = solver.vjp(struct('grad_x', [1; 2], 'grad_y', 1, 'solve_result', res));
            testCase.verifyEqual(grads_fr.status_val, 1)
            testCase.verifyEqual(grads_fr.c, grads.c, 'AbsTol', 1e-9)

            % solve result fields which are not real full double arrays are rejected
            bad = res;
            bad.x = single(res.x);
            testCase.verifyError(@() solver.vjp('grad_x', [1; 2], 'solve_result', bad), 'piqp:solve_result')
            bad = res;
            bad.z_l = complex(res.z_l);
            testCase.verifyError(@() solver.vjp('grad_x', [1; 2], 'solve_result', bad), 'piqp:solve_result')
            bad = res;
            bad.s_bl = sparse(res.s_bl);
            testCase.verifyError(@() solver.vjp('grad_x', [1; 2], 'solve_result', bad), 'piqp:solve_result')
            testCase.verifyError(@() solver.vjp('grad_x', [1; 2], 'solve_result', 1), 'piqp:solve_result')

            res.info.rho = 0;
            grads_fr = solver.vjp('grad_x', [1; 2], 'solve_result', res);
            testCase.verifyEqual(grads_fr.status_val, -10) % PIQP_INVALID_SETTINGS (invalid solve result)
        end

        function check_jvp(testCase, backend)
            solver = setup_vjp_solver(testCase, backend, testCase.c);
            tangents = solver.jvp('dc', [1; 2]);
            testCase.verifyEqual(tangents.status_val, -9) % PIQP_UNSOLVED (no solution)
            testCase.verifyEmpty(tangents.x)

            res = solver.solve();
            outputs = {'x', 'y', 'z_l', 'z_u', 'z_bl', 'z_bu', 's_l', 's_u', 's_bl', 's_bu'};

            % finite differences along c and along G, the matrix is passed in the format of the backend
            eps = 1e-6;
            dc = [0.3; -0.7];
            dG = [0.2 -0.1; 0.4 0.3];
            if strcmp(backend, 'sparse')
                dG_arg = sparse(dG);
            else
                dG_arg = dG;
            end
            cases = {{'dc', dc}, {'dG', dG_arg}};
            plus = {jvp_solve(testCase, backend, testCase.c + eps * dc, testCase.G), ...
                    jvp_solve(testCase, backend, testCase.c, testCase.G + eps * dG)};
            minus = {jvp_solve(testCase, backend, testCase.c - eps * dc, testCase.G), ...
                     jvp_solve(testCase, backend, testCase.c, testCase.G - eps * dG)};
            for k = 1:2
                tangents = solver.jvp(cases{k}{:});
                testCase.verifyEqual(tangents.status_val, 1) % PIQP_SOLVED
                for o = 1:length(outputs)
                    name = outputs{o};
                    fd = (plus{k}.(name) - minus{k}.(name)) / (2 * eps);
                    finite = ~isinf(plus{k}.(name)); % infinite slacks
                    testCase.verifyEqual(tangents.(name)(finite), fd(finite), 'AbsTol', 1e-5)
                    testCase.verifyTrue(all(tangents.(name)(~finite) == 0))
                end
            end

            % tangents at a provided solve result, the default call linearizes at the same point
            tangents = solver.jvp('dc', dc);
            tangents_fr = solver.jvp(struct('dc', dc, 'solve_result', res));
            testCase.verifyEqual(tangents_fr.status_val, 1)
            testCase.verifyEqual(tangents_fr.x, tangents.x, 'AbsTol', 1e-9)

            testCase.verifyError(@() solver.jvp('dc', [1; 2; 3]), ?MException)
            testCase.verifyError(@() solver.jvp('dc', dc, 'solve_result', 1), 'piqp:solve_result')
        end

        function res = jvp_solve(testCase, backend, c, G)
            solver = piqp(backend);
            solver.setup(testCase.P, c, testCase.A, testCase.b, ...
                         G, testCase.h_l, testCase.h_u, ...
                         testCase.x_l, [Inf; 0.1], ...
                         'eps_abs', 1e-12, 'eps_rel', 1e-12);
            res = solver.solve();
        end

        function L = vjp_loss(testCase, backend, c)
            solver = setup_vjp_solver(testCase, backend, c);
            res = solver.solve();
            L = res.x(1) + 2 * res.x(2) + res.y;
        end
    end

end