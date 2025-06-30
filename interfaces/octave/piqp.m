% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
% Copyright (c) 2017 Bartolomeo Stellato
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

classdef piqp < handle
    % piqp interface class for PIQP solver
    % This class provides a complete interface to the C++ implementation
    % of the PIQP solver.
    %
    % piqp Properties:
    %   objectHandle  - pointer to the C++ structure of PIQP solver
    %   isDense       - is dense or sparse backend used
    %   n             - number of variables
    %   p             - number of equality constraints
    %   m             - number of inequalty constraints
    %
    % piqp Methods:
    %
    %   setup             - configure solver with problem data
    %   solve             - solve the QP
    %   update            - modify problem data
    %
    %   get_settings      - get the current solver settings
    %   update_settings   - update the current solver settings
    %
    %   get_dimensions    - get the number of variables, equalty and
    %                       inequalty constraints
    %   version           - return PIQP version

    properties (SetAccess = private, Hidden = true)
        objectHandle % Handle to underlying C++ instance
    end

    properties (SetAccess = private)
        isDense = false
        n = 0
        p = 0
        m = 0
    end

    methods(Static)
        %%
        function out = version()
            % Return PIQP version
            out = piqp_oct('version');
        end
    end

    methods
        %%
        function this = piqp(varargin)
            % Construct piqp solver class.
            
            if length(varargin) >= 1
                if strcmp(varargin{1}, 'dense')
                    this.isDense = true;
                else
                    this.isDense = false;
                end
            else
                this.isDense = false;
            end

            this.objectHandle = piqp_oct('new', varargin{:});
        end

        %%
        function delete(this)
            % Destroy piqp solver class.
            piqp_oct('delete', this.objectHandle);
        end

        %%
        function out = get_settings(this)
            % GET_SETTINGS get the current solver settings structure.
            out = piqp_oct('get_settings', this.objectHandle);
        end

        %%
        function update_settings(this, varargin)
            % UPDATE_SETTINGS update the current solver settings structure.

            newSettings = validateSettings(this, varargin{:});

            %write the solver settings.  C-oct does not check input
            %data or protect against disallowed parameter modifications
            piqp_oct('update_settings', this.objectHandle, newSettings);
        end

        %%
        function [n,p,m] = get_dimensions(this)
            % GET_DIMENSIONS get the number of variables and constraints.

            [n,p,m] = piqp_oct('get_dimensions', this.objectHandle);
        end

        %%
        function setup(this, P, c, A, b, G, h_l, h_u, x_l, x_u, varargin)
            % SETUP Configure solver with problem data.
            %
            %   setup(P,c,A,b,G,h_l,h_u,x_l,x_u,options)

            % Get number of variables n
            if ~isempty(P)
                this.n = size(P, 1);
                assert(size(P, 2) == this.n, 'P must be square')
            else
                if ~isempty(c)
                    this.n = length(c);
                else
                    if ~isempty(A)
                        this.n = size(A, 2);
                    else
                        if ~isempty(G)
                            this.n = size(G, 2);
                        else
                            error('The problem does not have any variables');
                        end
                    end
                end
            end

            % Get number of equality constraints p
            if isempty(A)
                this.p = 0;
            else
                this.p = size(A, 1);
                assert(size(A, 2) == this.n, 'Incorrect dimension of A');
            end

            % Get number of inequality constraints m
            if isempty(G)
                this.m = 0;
            else
                this.m = size(G, 1);
                assert(size(G, 2) == this.n, 'Incorrect dimension of G');
            end

            %
            % Create dense/sparse matrices and full vectors if they are empty
            %
            if isempty(P)
                if this.isDense
                    P = zeros(this.n, this.n);
                else
                    P = sparse(this.n, this.n);
                end
            else
                if this.isDense
                    P = full(P);
                else
                    P = sparse(P);
                end
            end

            if isempty(c)
                c = zeros(this.n, 1);
            else
                c = full(c(:));
            end

            % Create proper equalty constraints if they are not passed
            if (isempty(A) && ~isempty(b)) || (~isempty(A) && isempty(b))
                error('A must be supplied together with b');
            end

            if isempty(A)
                if this.isDense
                    A = zeros(this.p, this.n);
                else
                    A = sparse(this.p, this.n);
                end
                b = zeros(this.p, 1);
            else
                if this.isDense
                    A = full(A);
                else
                    A = sparse(A);
                end
                b  = full(b(:));
            end

            % Create proper inequalty constraints if they are not passed
            if (isempty(G) && ~isempty([h_l; h_u])) || (~isempty(G) && isempty([h_l; h_u]))
                error('G must be supplied together with h');
            end

            if isempty(G)
                if this.isDense
                    G = zeros(this.m, this.n);
                else
                    G = sparse(this.m, this.n);
                end
                h_l = -Inf(this.m, 1);
                h_u = Inf(this.m, 1);
            else
                if this.isDense
                    G = full(G);
                else
                    G = sparse(G);
                end
                h_l  = full(h_l(:));
                h_u  = full(h_u(:));
            end

            if isempty(x_l)
                x_l = -Inf(this.n, 1);
            else
                x_l = full(x_l(:));
            end

            if isempty(x_u)
                x_u = Inf(this.n, 1);
            else
                x_u = full(x_u(:));
            end

            %
            % Check vector dimensions
            %
            assert(length(c) == this.n, 'Incorrect dimension of c');
            assert(length(b) == this.p, 'Incorrect dimension of b');
            assert(length(h_l) == this.m, 'Incorrect dimension of h_l');
            assert(length(h_u) == this.m, 'Incorrect dimension of h_u');
            assert(length(x_l) == this.n, 'Incorrect dimension of x_l');
            assert(length(x_u) == this.n, 'Incorrect dimension of x_u');

            %make a settings structure from the remainder of the arguments.
            settings = validateSettings(this, varargin{:});

            piqp_oct('setup',this.objectHandle,this.n,this.p,this.m,P,c,A,b,G,h_l,h_u,x_l,x_u,settings);
        end

        %%
        function res = solve(this, varargin)
            % SOLVE solve the QP.

            res = piqp_oct('solve', this.objectHandle);
        end

        %%
        function update(this, varargin)
            % UPDATE update problem data.

            assert(this.n ~= 0, 'Problem is not initialized.')

            allowedFields = {'P','c','A','b','G','h_l','h_u','x_l','x_u'};

            if isempty(varargin)
                return;
            elseif length(varargin) == 1
                if(~isstruct(varargin{1}))
                    error('Single input should be a structure with new problem data');
                else
                    newData = varargin{1};
                end
            else % param / value style assumed
                newData = struct(varargin{:});
            end

            %check for unknown fields
            newFields = fieldnames(newData);
            badFieldsIdx = find(~ismember(newFields,allowedFields));
            if(~isempty(badFieldsIdx))
                 error('Unrecognized input field ''%s'' detected',newFields{badFieldsIdx(1)});
            end

            if isfield(newData, 'P'); P = newData.P; else; P = []; end
            if isfield(newData, 'c'); c = newData.c; else; c = []; end
            if isfield(newData, 'A'); A = newData.A; else; A = []; end
            if isfield(newData, 'b'); b = newData.b; else; b = []; end
            if isfield(newData, 'G'); G = newData.G; else; G = []; end
            if isfield(newData, 'h_l'); h_l = newData.h_l; else; h_l = []; end
            if isfield(newData, 'h_u'); h_u = newData.h_u; else; h_u = []; end
            if isfield(newData, 'x_l'); x_l = newData.x_l; else; x_l = []; end
            if isfield(newData, 'x_u'); x_u = newData.x_u; else; x_u = []; end

            if ~isempty(P)
                if this.isDense
                    P = full(P);
                else
                    P = sparse(P);
                end
                assert(size(P, 1) == this.n && size(P, 2) == this.n, 'Incorrect dimension of P')
            end

            if ~isempty(c)
                c = full(c(:));
                assert(length(c) == this.n, 'Incorrect dimension of c');
            end

            if ~isempty(A)
                if this.isDense
                    A = full(A);
                else
                    A = sparse(A);
                end
                assert(size(A, 1) == this.p && size(A, 2) == this.n, 'Incorrect dimension of A')
            end

            if ~isempty(b)
                b = full(b(:));
                assert(length(b) == this.p, 'Incorrect dimension of b');
            end

            if ~isempty(G)
                if this.isDense
                    G = full(G);
                else
                    G = sparse(G);
                end
                assert(size(G, 1) == this.m && size(G, 2) == this.n, 'Incorrect dimension of G')
            end

            if ~isempty(h_l)
                h_l = full(h_l(:));
                assert(length(h_l) == this.m, 'Incorrect dimension of h_l');
            end

            if ~isempty(h_u)
                h_u = full(h_u(:));
                assert(length(h_u) == this.m, 'Incorrect dimension of h_u');
            end

            if ~isempty(x_l)
                x_l = full(x_l(:));
                assert(length(x_l) == this.n, 'Incorrect dimension of x_l');
            end

            if ~isempty(x_u)
                x_u = full(x_u(:));
                assert(length(x_u) == this.n, 'Incorrect dimension of x_u');
            end

            piqp_oct('update',this.objectHandle,this.n,this.p,this.m,P,c,A,b,G,h_l,h_u,x_l,x_u);
        end
    end
end

function settings = validateSettings(this, varargin)

settings = piqp_oct('get_settings', this.objectHandle);

%no settings passed -> return defaults
if isempty(varargin)
    return;
end

%check for structure style input
if isstruct(varargin{1})
    newSettings = varargin{1};
    assert(length(varargin) == 1, 'too many input arguments');
else
    newSettings = struct(varargin{:});
end

%get the piqp settings fields
currentFields = fieldnames(settings);

%get the requested fields in the update
newFields = fieldnames(newSettings);

%check for unknown parameters
badFieldsIdx = find(~ismember(newFields, currentFields));
if ~isempty(badFieldsIdx)
    error('Unrecognized solver setting ''%s'' detected', newFields{badFieldsIdx(1)});
end

%everything checks out - merge the newSettings into the current ones
for i = 1:length(newFields)
    settings.(newFields{i}) = double(newSettings.(newFields{i}));
end

end
