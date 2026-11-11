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

        function backward_dense_qp(testCase)
            check_backward(testCase, 'dense');
        end

        function backward_sparse_qp(testCase)
            check_backward(testCase, 'sparse');
        end

        function backward_not_enabled(testCase)
            testCase.solver_dense.solve();
            grads = testCase.solver_dense.backward('grad_x', [1; 2]);
            testCase.verifyEqual(grads.status_val, -10) % PIQP_INVALID_SETTINGS
            testCase.verifyEmpty(grads.c)
        end
    end

    methods
        function solver = setup_backward_solver(testCase, backend, c)
            % same problem with an active upper bound x2 <= 0.1
            solver = piqp(backend);
            solver.setup(testCase.P, c, testCase.A, testCase.b, ...
                         testCase.G, testCase.h_l, testCase.h_u, ...
                         testCase.x_l, [Inf; 0.1], ...
                         'eps_abs', 1e-12, 'eps_rel', 1e-12, 'enable_backward', true);
        end

        function check_backward(testCase, backend)
            % loss L = x1 + 2 x2 + y
            solver = setup_backward_solver(testCase, backend, testCase.c);
            grads = solver.backward('grad_x', [1; 2], 'grad_y', 1);
            testCase.verifyEqual(grads.status_val, -9) % PIQP_UNSOLVED (no forward solution)
            testCase.verifyEmpty(grads.c)

            res = solver.solve();
            grads = solver.backward('grad_x', [1; 2], 'grad_y', 1);
            testCase.verifyEqual(grads.status_val, 1) % PIQP_SOLVED

            % finite differences w.r.t. c
            eps = 1e-6;
            for i = 1:2
                dc = zeros(2, 1);
                dc(i) = eps;
                fd = (backward_loss(testCase, backend, testCase.c + dc) - ...
                      backward_loss(testCase, backend, testCase.c - dc)) / (2 * eps);
                testCase.verifyEqual(grads.c(i), fd, 'AbsTol', 1e-5)
            end
            testCase.verifyEqual(size(grads.P), [2 2])
            testCase.verifyEqual(size(grads.A), [1 2])
            testCase.verifyEqual(size(grads.G), [2 2])
            testCase.verifyEqual(issparse(grads.A), strcmp(backend, 'sparse'))

            % gradients at a provided forward result, the default call linearizes at the same point
            grads_fr = solver.backward(struct('grad_x', [1; 2], 'grad_y', 1, 'forward_result', res));
            testCase.verifyEqual(grads_fr.status_val, 1)
            testCase.verifyEqual(grads_fr.c, grads.c, 'AbsTol', 1e-9)

            % forward result fields which are not real full double arrays are rejected
            bad = res;
            bad.x = single(res.x);
            testCase.verifyError(@() solver.backward('grad_x', [1; 2], 'forward_result', bad), 'piqp:backward')
            bad = res;
            bad.z_l = complex(res.z_l);
            testCase.verifyError(@() solver.backward('grad_x', [1; 2], 'forward_result', bad), 'piqp:backward')
            bad = res;
            bad.s_bl = sparse(res.s_bl);
            testCase.verifyError(@() solver.backward('grad_x', [1; 2], 'forward_result', bad), 'piqp:backward')
            testCase.verifyError(@() solver.backward('grad_x', [1; 2], 'forward_result', 1), 'piqp:backward')

            res.info.rho = 0;
            grads_fr = solver.backward('grad_x', [1; 2], 'forward_result', res);
            testCase.verifyEqual(grads_fr.status_val, -10) % PIQP_INVALID_SETTINGS (invalid forward result)
        end

        function L = backward_loss(testCase, backend, c)
            solver = setup_backward_solver(testCase, backend, c);
            res = solver.solve();
            L = res.x(1) + 2 * res.x(2) + res.y;
        end
    end

end