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
    end

end